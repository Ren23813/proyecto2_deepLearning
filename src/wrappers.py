"""
wrappers.py
-----------
Construcción del entorno preprocesado para entrenar/evaluar agentes de Deep
Q-Learning sobre ALE/SpaceInvaders-v5.


- gymnasium.wrappers.AtariPreprocessing: escala de grises, redimensionamiento
  a 84x84, "max-pool" sobre los últimos 2 frames dentro de cada bloque de
  frame skipping, y (opcionalmente) fin de episodio al perder una vida.
- gymnasium.wrappers.FrameStackObservation: apila los últimos `stack_size`
  frames preprocesados para darle al agente información de movimiento.
- ClipRewardWrapper (definido aquí): recorta la recompensa a {-1, 0, 1},
  como en el paper original de DQN, para estabilizar el entrenamiento.
  Se aplica solo durante entrenamiento, nunca durante evaluación/competencia,
  ya que la competencia se basa en la recompensa real acumulada.

Nota:
    ALE/SpaceInvaders-v5 debe crearse con frameskip=1 y repeat_action_probability
    explícito, porque AtariPreprocessing implementa su propio frame skipping
    (con max-pooling de los últimos 2 frames) y espera un entorno base sin
    frame skipping propio.
"""

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class ClipRewardWrapper(gym.RewardWrapper):
    """Recorta la recompensa de cada paso a su signo: {-1, 0, +1}.
    """

    def reward(self, reward):
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -1.0
        return 0.0


def build_atari_env(
    env_id="ALE/SpaceInvaders-v5",
    render_mode="rgb_array",
    frame_skip=4,
    screen_size=84,
    stack_size=4,
    terminal_on_life_loss=True,
    clip_rewards=True,
    repeat_action_probability=0.25,
    full_action_space=False,
    noop_max=30,
    video_folder=None,
    name_prefix="video",
    episode_trigger=None,
):
    """
    Crea el entorno de Space Invaders con el stack de preprocesamiento
    estándar para Deep Q-Learning.

    Parámetros
    ----------
    env_id : str
        Id del entorno base de ALE.
    render_mode : str
        Debe ser "rgb_array" para poder grabar video con RecordVideo; el
        renderizado usa siempre los frames RGB originales del emulador,
        independientemente del preprocesamiento aplicado a las observaciones
        que ve el agente.
    frame_skip : int
        Número de frames del emulador que se repiten por cada acción
        (aplicado dentro de AtariPreprocessing, no en el entorno base).
    screen_size : int
        Tamaño (screen_size x screen_size) al que se redimensiona cada frame
        en escala de grises.
    stack_size : int
        Número de frames preprocesados consecutivos que se apilan para
        formar la observación final que recibe la red.
    terminal_on_life_loss : bool
        Si True, el episodio (desde la perspectiva del agente/entrenamiento)
        termina cada vez que se pierde una vida, no solo cuando termina la
        partida completa. Ayuda a que el agente aprenda más rápido a evitar
        perder vidas. Debe ponerse en False para evaluación/competencia,
        donde interesa el puntaje de la partida completa (con sus 3 vidas).
    clip_rewards : bool
        Si True, aplica ClipRewardWrapper. Debe ser False en evaluación.
    repeat_action_probability : float
        Probabilidad de "sticky actions" del entorno base de ALE: fuente de
        estocasticidad 0.25 
    full_action_space : bool
        Si True, usa las 18 acciones del joystick de Atari; si False
        (default), usa solo las 6 acciones relevantes para Space Invaders.
    noop_max : int
        Número máximo de acciones NOOP aleatorias al inicio de cada episodio
        (introduce variabilidad en el estado inicial).
    video_folder, name_prefix, episode_trigger :
        Igual que en ale_utils.crear_entorno del Laboratorio #5: si
        video_folder no es None, se envuelve el entorno final con
        gymnasium.wrappers.RecordVideo.

    Retorna
    -------
    env : gymnasium.Env
        Entorno listo para usarse con un agente DQN. Las observaciones
        tienen forma (stack_size, screen_size, screen_size), dtype uint8.
    """
    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,  # el frame-skip real lo hace AtariPreprocessing
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
    )

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        scale_obs=False, 
    )

    if clip_rewards:
        env = ClipRewardWrapper(env)

    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)

    if video_folder is not None:
        import os
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
        )

    return env


def build_train_env(env_id="ALE/SpaceInvaders-v5", **kwargs):
    """Entorno configurado para entrenamiento: reward clipping activado y
    fin de episodio al perder una vida (terminal_on_life_loss=True)."""
    defaults = dict(clip_rewards=True, terminal_on_life_loss=True, video_folder=None)
    defaults.update(kwargs)
    return build_atari_env(env_id, **defaults)


def build_eval_env(env_id="ALE/SpaceInvaders-v5", video_folder=None,
                    name_prefix="eval", **kwargs):
    """Entorno configurado para evaluación/competencia: sin reward clipping
    y sin terminar el episodio al perder una vida, para reportar el puntaje
    real de la partida completa (3 vidas)."""
    defaults = dict(clip_rewards=False, terminal_on_life_loss=False)
    defaults.update(kwargs)
    return build_atari_env(
        env_id, video_folder=video_folder, name_prefix=name_prefix, **defaults
    )
