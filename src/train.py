"""
train.py
--------
Loop de entrenamiento de un agente Dueling Double DQN sobre
ALE/SpaceInvaders-v5, con soporte de:

- N-step returns (--n-step): acelera la propagación de la señal de
  recompensa usando retornos de n pasos en vez de bootstrapping de 1 paso.
- Actualización de la red target "dura" (hard, por defecto, como en el
  paper original de DQN) o "suave" (soft / Polyak averaging, --target-update-mode soft),
  que suele dar entrenamientos más estables.
- Reanudar entrenamiento (--resume-from) desde un checkpoint guardado
  previamente, indispensable para entrenar en sesiones de Google Colab que
  se desconectan. NOTA: el replay buffer NO se persiste entre sesiones
  (sería demasiado grande para guardar/cargar rápidamente); al resumir se
  vuelve a llenar desde cero durante `--learning-starts` pasos, pero el
  contador de pasos totales, epsilon y los pesos/optimizador sí continúan
  donde se quedaron.
- Evaluación periódica (--eval-every) con política greedy sobre el
  entorno de evaluación (SIN reward clipping, SIN terminal_on_life_loss),
  para llevar un registro del puntaje REAL a lo largo del entrenamiento,
  no solo de la recompensa clippeada de entrenamiento.

Uso típico (GPU recomendada, ver README.md):

    python train.py --total-steps 8000000 --run-name corrida02 \\
                     --n-step 3 --target-update-mode soft --eval-every 200000

Para resumir una corrida interrumpida:

    python train.py --resume-from ../checkpoints/corrida02_latest.pt \\
                     --run-name corrida02 --total-steps 8000000 ...
    (usa los MISMOS argumentos de la corrida original)
"""

import argparse
import csv
import os
import shutil
import time
from collections import deque

import numpy as np

from wrappers import build_train_env, build_eval_env
from agent import DQNAgent
from replay_buffer import ReplayBuffer


def linear_epsilon(step, eps_start, eps_end, decay_steps):
    if step >= decay_steps:
        return eps_end
    frac = step / decay_steps
    return eps_start + frac * (eps_end - eps_start)


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento Dueling Double DQN - Space Invaders")
    p.add_argument("--env-id", default="ALE/SpaceInvaders-v5")
    p.add_argument("--run-name", default="run01")
    p.add_argument("--total-steps", type=int, default=2_000_000,
                   help="Pasos de entorno (frames tras frame-skip) a entrenar en TOTAL "
                        "(incluyendo los ya hechos si se usa --resume-from).")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-steps", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=50_000,
                   help="Pasos de calentamiento con política aleatoria antes de entrenar "
                        "(también aplica tras --resume-from, para rellenar el buffer).")
    p.add_argument("--train-freq", type=int, default=4,
                   help="Entrenar cada N pasos de entorno.")
    p.add_argument("--n-step", type=int, default=1,
                   help="Número de pasos para el retorno n-step (1 = DQN clásico de 1 paso).")
    p.add_argument("--target-update-mode", choices=["hard", "soft"], default="hard")
    p.add_argument("--target-update-freq", type=int, default=10_000,
                   help="[modo hard] cada cuántos pasos se copia online_net -> target_net.")
    p.add_argument("--tau", type=float, default=0.005,
                   help="[modo soft] coeficiente de Polyak averaging por paso de entrenamiento.")
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=0,
                   help="Cada cuántos pasos correr una evaluación greedy (0 = desactivado).")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--resume-from", default=None,
                   help="Ruta a un checkpoint .pt para resumir el entrenamiento.")
    p.add_argument("--resume-step-override", type=int, default=None,
                   help="Fuerza manualmente el 'step' de partida al resumir, por si el "
                        "checkpoint fue guardado con una versión anterior de train.py que no "
                        "almacenaba el step/episodio dentro del archivo (en ese caso, sin este "
                        "flag, se asume step=0). Útil también para corregir manualmente el "
                        "conteo de epsilon tras una recuperación de emergencia.")
    p.add_argument("--drive-sync-dir", default=None,
                   help="Si se especifica (p. ej. una carpeta en Google Drive montado), "
                        "copia ahí el checkpoint 'latest' y los CSV de log cada vez que se "
                        "guarda un checkpoint. Se recomienda mantener --log-dir y "
                        "--checkpoint-dir en disco LOCAL de Colab (rápido) y usar este flag "
                        "para respaldar en Drive solo periódicamente, en vez de escribir "
                        "directamente en Drive en cada episodio (eso puede colgarse por la "
                        "latencia de la capa FUSE de Drive).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="../logs")
    p.add_argument("--checkpoint-dir", default="../checkpoints")
    return p.parse_args()


def evaluar_greedy(agent, env_id, n_episodios, max_steps=100_000):
    """Evaluación rápida con política greedy sobre el entorno SIN reward
    clipping y SIN terminal_on_life_loss, para medir el puntaje real."""
    env = build_eval_env(env_id)
    recompensas = []
    try:
        for _ in range(n_episodios):
            obs, info = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = agent.act_greedy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            recompensas.append(total)
    finally:
        env.close()
    return recompensas


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.run_name}.csv")
    eval_log_path = os.path.join(args.log_dir, f"{args.run_name}_eval.csv")

    env = build_train_env(args.env_id, render_mode="rgb_array")
    n_actions = env.action_space.n

    agent = DQNAgent(n_actions=n_actions, lr=args.lr, gamma=args.gamma, seed=args.seed)
    print("Dispositivo de entrenamiento:", agent.device)

    buffer = ReplayBuffer(capacity=args.buffer_size, obs_shape=(4, 84, 84))

    start_step = 0
    episode_idx = 0

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = agent.load(args.resume_from, load_optimizer=True)
        start_step = checkpoint.get("step", 0)
        episode_idx = checkpoint.get("episode_idx", 0)
        if args.resume_step_override is not None:
            print(f"(Se ignora el step guardado en el checkpoint ({start_step}); "
                  f"se fuerza --resume-step-override={args.resume_step_override})")
            start_step = args.resume_step_override
        print(f"Resumiendo desde {args.resume_from}: step={start_step}, "
              f"episodio={episode_idx}. El replay buffer se vuelve a llenar "
              f"desde cero durante los próximos --learning-starts pasos.")
        write_header = not os.path.exists(log_path)
    else:
        write_header = True

    if write_header:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "episode", "episode_reward", "episode_length",
                              "epsilon", "avg_loss_last_train", "elapsed_seconds"])

    if args.eval_every > 0 and not os.path.exists(eval_log_path):
        with open(eval_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward_mean", "reward_max", "rewards"])

    # --- n-step returns: buffer temporal de las últimas n transiciones ---
    n_step_buffer = deque(maxlen=args.n_step)

    def flush_n_step_transition(force_all=False):
        """Calcula y guarda en el replay buffer la transición n-step más
        antigua en n_step_buffer, si ya hay suficientes pasos acumulados
        (o si force_all=True, para vaciar al terminar un episodio)."""
        while len(n_step_buffer) == args.n_step or (force_all and len(n_step_buffer) > 0):
            obs0, action0, _, _, _ = n_step_buffer[0]
            R = 0.0
            done_n = False
            next_obs_n = n_step_buffer[-1][3]
            for i, (_, _, r_i, _, done_i) in enumerate(n_step_buffer):
                R += (args.gamma ** i) * r_i
                if done_i:
                    done_n = True
                    next_obs_n = n_step_buffer[i][3]
                    break
            buffer.push(obs0, action0, R, next_obs_n, done_n)
            n_step_buffer.popleft()
            if not force_all:
                break

    obs, info = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    losses = []
    t0 = time.time()
    step = start_step

    def guardar_emergencia():
        emergency_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_latest.pt")
        agent.save(emergency_path, extra={"step": step, "episode_idx": episode_idx})
        print(f"Checkpoint de emergencia guardado en {emergency_path}. "
              f"Puedes resumir con --resume-from {emergency_path}")
        if args.drive_sync_dir:
            try:
                os.makedirs(args.drive_sync_dir, exist_ok=True)
                shutil.copy2(emergency_path, os.path.join(
                    args.drive_sync_dir, os.path.basename(emergency_path)))
                print(f"Checkpoint de emergencia también respaldado en {args.drive_sync_dir}")
            except Exception as e:
                print(f"[aviso] no se pudo sincronizar el checkpoint de emergencia a Drive: {e}")

    try:
        for step in range(start_step + 1, args.total_steps + 1):
            epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
            action = agent.select_action(obs, epsilon)
    
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
            n_step_buffer.append((obs, action, reward, next_obs, done))
            flush_n_step_transition()
    
            obs = next_obs
            episode_reward += reward
            episode_length += 1
    
            if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
                batch = buffer.sample(args.batch_size)
                loss = agent.train_step(batch)
                losses.append(loss)
    
            if args.target_update_mode == "hard":
                if step % args.target_update_freq == 0:
                    agent.update_target_network()
            else:  # soft
                if len(buffer) >= args.learning_starts:
                    agent.soft_update_target_network(tau=args.tau)
    
            if done:
                flush_n_step_transition(force_all=True)
    
                avg_loss = float(np.mean(losses)) if losses else float("nan")
                elapsed = time.time() - t0
                print(f"[step {step}] episodio {episode_idx} | "
                      f"recompensa={episode_reward:.1f} | pasos={episode_length} | "
                      f"epsilon={epsilon:.3f} | loss={avg_loss:.4f} | t={elapsed:.0f}s")
    
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, episode_idx, episode_reward, episode_length,
                                      round(epsilon, 4), avg_loss, round(elapsed, 1)])
    
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_idx += 1
                losses = []
    
            if args.eval_every > 0 and step % args.eval_every == 0:
                recompensas_eval = evaluar_greedy(agent, args.env_id, args.eval_episodes)
                r_mean = float(np.mean(recompensas_eval))
                r_max = float(np.max(recompensas_eval))
                print(f"  [EVAL greedy @ step {step}] promedio={r_mean:.1f} "
                      f"máximo={r_max:.1f} episodios={recompensas_eval}")
                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, r_mean, r_max, recompensas_eval])
    
            if step % args.checkpoint_freq == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_step{step}.pt")
                agent.save(ckpt_path, extra={"step": step, "episode_idx": episode_idx})
                # "latest" siempre apunta al checkpoint más reciente, para
                # simplificar el --resume-from (no hay que recordar el número
                # de step exacto en el nombre del archivo).
                latest_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_latest.pt")
                agent.save(latest_path, extra={"step": step, "episode_idx": episode_idx})
                print(f"  checkpoint guardado en {ckpt_path}")
    
                if args.drive_sync_dir:
                    try:
                        os.makedirs(args.drive_sync_dir, exist_ok=True)
                        shutil.copy2(latest_path, os.path.join(
                            args.drive_sync_dir, os.path.basename(latest_path)))
                        if os.path.exists(log_path):
                            shutil.copy2(log_path, os.path.join(
                                args.drive_sync_dir, os.path.basename(log_path)))
                        if os.path.exists(eval_log_path):
                            shutil.copy2(eval_log_path, os.path.join(
                                args.drive_sync_dir, os.path.basename(eval_log_path)))
                        print(f"  respaldo sincronizado a {args.drive_sync_dir}")
                    except Exception as e:
                        # Un fallo de sincronización a Drive NUNCA debe tumbar
                        # el entrenamiento: solo se avisa y se continúa, ya que
                        # el checkpoint local (rápido) ya quedó guardado.
                        print(f"  [aviso] no se pudo sincronizar a Drive: {e}")

    except KeyboardInterrupt:
        print("\nInterrupción manual detectada (Ctrl+C). Guardando checkpoint de emergencia...")
        guardar_emergencia()
        env.close()
        return

    final_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_final.pt")
    agent.save(final_path, extra={"step": args.total_steps, "episode_idx": episode_idx})
    print("Checkpoint final guardado en", final_path)

    if args.drive_sync_dir:
        try:
            os.makedirs(args.drive_sync_dir, exist_ok=True)
            shutil.copy2(final_path, os.path.join(
                args.drive_sync_dir, os.path.basename(final_path)))
            print(f"Checkpoint final también respaldado en {args.drive_sync_dir}")
        except Exception as e:
            print(f"[aviso] no se pudo sincronizar el checkpoint final a Drive: {e}")

    env.close()


if __name__ == "__main__":
    main()
