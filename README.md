# Proyecto 2: Competencia de Agentes en Space Invaders (Dueling Double DQN)


Agente de Reinforcement Learning entrenado con Dueling Double DQN para
jugar `ALE/SpaceInvaders-v5` (Gymnasium + ale-py). Este repositorio contiene
todo el código de preprocesamiento, entrenamiento, evaluación y generación
de video usado en el proyecto.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── wrappers.py        # Preprocesamiento del entorno (AtariPreprocessing, FrameStack, reward clipping)
│   ├── model.py            # Red neuronal Dueling DQN (PyTorch)
│   ├── replay_buffer.py    # Buffer de repetición de experiencias
│   ├── agent.py             # Lógica de Double DQN (selección de acción, entrenamiento, save/load)
│   ├── train.py             # Script de entrenamiento (CLI)
│   └── evaluate.py          # Script de evaluación / competencia (CLI)
├── notebooks/
│   └── Proyecto2_DQN_SpaceInvaders.ipynb   # Notebook de demostración end-to-end
├── checkpoints/             # Pesos guardados (.pt)
├── logs/                    # CSV con historial de entrenamiento por episodio
├── videos/                  # Videos .mp4 generados por evaluate.py
└── report/
    └── plantilla_reporte.md # Plantilla para el trabajo escrito (secciones 2.1–2.6)
```

## Instalación

### Local: 
```bash
python -m venv venv
venv\Scripts\activate   #(Windows)
pip install -r requirements.txt
```

Asegurarse de instalar la versión de Pytorch de GPU. Sino, va a ir muy lento. 

## Cómo entrenar

```bash
cd src
python train.py --run-name corrida01 --total-steps 2000000
```

Parámetros más relevantes (ver `python train.py --help` para la lista
completa):

| Parámetro | Default | Descripción |
|---|---|---|
| `--total-steps` | 2,000,000 | Pasos de entorno (tras frame-skip) a entrenar |
| `--buffer-size` | 100,000 | Capacidad del replay buffer |
| `--batch-size` | 32 | Tamaño de batch de entrenamiento |
| `--lr` | 1e-4 | Tasa de aprendizaje (Adam) |
| `--gamma` | 0.99 | Factor de descuento |
| `--eps-start` / `--eps-end` / `--eps-decay-steps` | 1.0 / 0.1 / 1,000,000 | Programa de epsilon-greedy (decaimiento lineal) |
| `--learning-starts` | 50,000 | Pasos de calentamiento con política aleatoria antes de entrenar |
| `--train-freq` | 4 | Entrenar cada N pasos de entorno |
| `--target-update-freq` | 10,000 | Cada cuántos pasos se actualiza la red target (hard update) |
| `--checkpoint-freq` | 100,000 | Cada cuántos pasos se guarda un checkpoint intermedio |

El entrenamiento genera:
- `logs/<run_name>.csv`: una fila por episodio, con recompensa (clipped),
  longitud del episodio, epsilon y pérdida promedio. Úsalo para graficar
  las curvas de entrenamiento pedidas en la sección 2.3 del reporte.
- `checkpoints/<run_name>_step<N>.pt` y `checkpoints/<run_name>_final.pt`:
  pesos de las redes online y target.

### Tiempos de referencia: 
Tarda como hora y media en entrenar (local), usando GPU. 

### Entrenar en Google Colab

1. Sube la carpeta `src/` a Colab (o clona el repositorio).
2. Activa GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`.
3. Instala dependencias: `!pip install gymnasium ale-py`.
4. Ejecuta `!python src/train.py --run-name colab01 --total-steps 2000000`.
5. Descarga `checkpoints/colab01_final.pt` y `logs/colab01.csv` a tu máquina
   (o guarda en Google Drive) para continuar el análisis localmente.

## Cómo evaluar / generar el video 

```bash
cd src
python evaluate.py --checkpoint ../checkpoints/corrida01_final.pt --n-episodios 5 --video-folder ../videos/eval_corrida01
```

Esto ejecuta 5 episodios con política greedy (sin exploración) sobre el
entorno de evaluación (sin reward clipping), imprime la recompensa
de cada episodio, reporta el promedio y el máximo (criterio oficial de la
competencia), y guarda un video `.mp4` por episodio en la carpeta indicada.


## Cargar el modelo entrenado desde cero

```python
from wrappers import build_eval_env
from agent import DQNAgent

env = build_eval_env("ALE/SpaceInvaders-v5")
agent = DQNAgent(n_actions=env.action_space.n)
agent.load("../checkpoints/corrida01_final.pt")

obs, info = env.reset()
action = agent.act_greedy(obs)
```

## Notebook de demostración

`notebooks/Proyecto2_DQN_SpaceInvaders.ipynb` contiene una corrida corta
(pequeño test) de todo el pipeline (entrenamiento breve, curvas de
pérdida/recompensa, evaluación y video), pensada para verificar que el
código funciona de extremo a extremo antes de lanzar el entrenamiento largo
real en GPU. No representa el desempeño final del agente, para eso hay
que entrenar con los pasos indicados arriba.

## Decisiones de diseño (resumen; ver reporte para el detalle completo)

- Observación: escala de grises, 84x84, 4 frames apilados
  (`AtariPreprocessing` + `FrameStackObservation`), siguiendo Mnih et al. (2015).
- Acciones: espacio mínimo de 6 acciones (`full_action_space=False`).
- Recompensa: clipping a {-1,0,1} solo en entrenamiento; en evaluación se
  usa la recompensa real para reportar el puntaje de la competencia.
- Vidas: `terminal_on_life_loss=True` en entrenamiento (episodios más
  cortos y señal de aprendizaje más frecuente); `False` en evaluación
  (puntaje de la partida completa).
- Arquitectura: CNN de 3 capas convolucionales + Dueling heads
  (Value/Advantage), ver `model.py`.
- Algoritmo: Double DQN (selección con red online, evaluación con red
  target) para reducir sobreestimación de valores Q.
