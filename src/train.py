"""
train.py
--------
Loop de entrenamiento de un agente Dueling Double DQN sobre
ALE/SpaceInvaders-v5.

Uso típico (GPU recomendada):

    python train.py --total-steps 2000000 --run-name corrida01

Registra en un CSV (logs/<run_name>.csv) la recompensa y longitud de cada
episodio, además de la pérdida promedio y epsilon vigente, para poder
graficar curvas de entrenamiento y documentar iteraciones en el reporte
escrito (sección 2.3 del proyecto).

Guarda checkpoints periódicos en checkpoints/<run_name>_step<N>.pt y un
checkpoint final checkpoints/<run_name>_final.pt.
"""

import argparse
import csv
import os
import time

import numpy as np

from wrappers import build_train_env
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
                   help="Pasos de entorno (frames tras frame-skip) a entrenar.")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-steps", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=50_000,
                   help="Pasos de calentamiento con política aleatoria antes de entrenar.")
    p.add_argument("--train-freq", type=int, default=4,
                   help="Entrenar cada N pasos de entorno.")
    p.add_argument("--target-update-freq", type=int, default=10_000,
                   help="Cada cuántos pasos se copia online_net -> target_net.")
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="../logs")
    p.add_argument("--checkpoint-dir", default="../checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.run_name}.csv")

    env = build_train_env(args.env_id, render_mode="rgb_array")
    n_actions = env.action_space.n

    agent = DQNAgent(n_actions=n_actions, lr=args.lr, gamma=args.gamma, seed=args.seed)
    print("Dispositivo de entrenamiento:", agent.device)

    buffer = ReplayBuffer(capacity=args.buffer_size, obs_shape=(4, 84, 84))

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "episode_reward", "episode_length",
                          "epsilon", "avg_loss_last_train", "elapsed_seconds"])

    obs, info = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0
    losses = []
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        episode_reward += reward
        episode_length += 1

        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size)
            loss = agent.train_step(batch)
            losses.append(loss)

        if step % args.target_update_freq == 0:
            agent.update_target_network()

        if done:
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

        if step % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_step{step}.pt")
            agent.save(ckpt_path)
            print(f"  checkpoint guardado en {ckpt_path}")

    final_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_final.pt")
    agent.save(final_path)
    print("Checkpoint final guardado en", final_path)

    env.close()


if __name__ == "__main__":
    main()
