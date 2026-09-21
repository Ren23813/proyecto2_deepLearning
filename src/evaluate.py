"""
evaluate.py
-----------
Carga los pesos entrenados, ejecuta N episodios de evaluación con política greedy (epsilon=0, sin
exploración) sobre el entorno de evaluación (sin reward clipping, con las
3 vidas completas), reporta la recompensa de cada episodio y el máximo, y genera el video
del mejor episodio (o de todos, según --grabar-todos).

Reutiliza directamente el patrón de funciones del Laboratorio #5
(crear_entorno / ejecutar_episodio / generar_video_agente), adaptando la
función de agente para usar la política aprendida (agent.act_greedy) en
lugar del agente aleatorio o de regla simple.

Uso:
    python evaluate.py --checkpoint ../checkpoints/run01_final.pt \\
                        --n-episodios 5 --video-folder ../videos/eval_run01

    python src/evaluate.py --checkpoint checkpoints/corrida02_final.pt --n-episodios 5 --video-folder videos/eval_corrida02_v2
"""

import argparse
import os

import numpy as np

from wrappers import build_eval_env
from agent import DQNAgent


def ejecutar_episodio_evaluacion(env, agent, max_steps=100_000):
    """Análogo a ejecutar_episodio() del Lab, pero usando la
    política greedy del agente entrenado en vez de una función de agente
    aleatorio o de regla simple."""
    observation, info = env.reset()
    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = agent.act_greedy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        recompensa_total += reward
        pasos += 1
        if terminated or truncated:
            break

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": terminated,
        "truncated": truncated,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluación del agente entrenado - Space Invaders")
    p.add_argument("--checkpoint", required=True, help="Ruta al archivo .pt del agente entrenado.")
    p.add_argument("--env-id", default="ALE/SpaceInvaders-v5")
    p.add_argument("--n-episodios", type=int, default=5)
    p.add_argument("--video-folder", default="../videos/evaluacion")
    p.add_argument("--name-prefix", default="space_invaders_dqn")
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()

    # Un solo entorno, grabando todos los episodios de evaluación (así se
    # tiene el video del episodio de mayor puntaje sin tener que adivinar
    # cuál será antes de ejecutar).
    env = build_eval_env(
        args.env_id,
        video_folder=args.video_folder,
        name_prefix=args.name_prefix,
        episode_trigger=lambda ep: True,
    )

    n_actions = env.action_space.n
    agent = DQNAgent(n_actions=n_actions)
    agent.load(args.checkpoint)
    print(f"Pesos cargados desde: {args.checkpoint}")
    print(f"Dispositivo: {agent.device}")

    recompensas = []
    try:
        for ep in range(args.n_episodios):
            resultado = ejecutar_episodio_evaluacion(env, agent, max_steps=args.max_steps)
            recompensas.append(resultado["recompensa_total"])
            print(f"Episodio {ep}: pasos={resultado['pasos']}, "
                  f"recompensa_total={resultado['recompensa_total']}")
    finally:
        env.close()  

    recompensas = np.array(recompensas)
    print()
    print("=" * 50)
    print(f"Recompensas de los {args.n_episodios} episodios: {recompensas.tolist()}")
    print(f"Recompensa promedio: {recompensas.mean():.2f}")
    print(f"Recompensa máxima (criterio de competencia): {recompensas.max():.2f}")
    print("=" * 50)

    videos = []
    if os.path.isdir(args.video_folder):
        for archivo in sorted(os.listdir(args.video_folder)):
            if archivo.startswith(args.name_prefix) and archivo.endswith(".mp4"):
                videos.append(os.path.join(args.video_folder, archivo))
    print("Videos generados:")
    for v in videos:
        print("-", v)


if __name__ == "__main__":
    main()
