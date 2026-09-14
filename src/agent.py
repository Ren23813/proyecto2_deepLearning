"""
agent.py
--------
Agente de Double DQN + Dueling DQN.

- "Dueling": la arquitectura de la red (ver model.py) descompone Q(s,a) en
  V(s) + A(s,a), lo que ayuda a aprender el valor de un estado sin tener que
  aprender el efecto de cada acción por separado (útil en Space Invaders,
  donde muchos estados tienen un valor similar sin importar la acción, ej.
  cuando no hay ningún disparo enemigo cerca).

- "Double" DQN: para calcular el objetivo de
  entrenamiento, la ACCIÓN se selecciona con la red online (la que se está
  entrenando) pero su VALOR se evalúa con la red target. Esto corrige el
  sesgo de sobreestimación de valores Q que tiene el DQN clásico (donde la
  misma red target selecciona y evalúa la acción, tendiendo a elegir
  sobreestimaciones ruidosas).

Fórmula del objetivo (Double DQN):
    a* = argmax_a Q_online(s', a)
    y  = r + gamma * (1 - done) * Q_target(s', a*)
    loss = Huber( Q_online(s, a) , y )
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DuelingDQN


class DQNAgent:
    def __init__(
        self,
        n_actions,
        in_channels=4,
        lr=1e-4,
        gamma=0.99,
        device=None,
        seed=0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.online_net = DuelingDQN(in_channels, n_actions).to(self.device)
        self.target_net = DuelingDQN(in_channels, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def select_action(self, observation, epsilon):
        """Política epsilon-greedy sobre la red online.

        Parámetros
        ----------
        observation : np.ndarray, forma (4, 84, 84), uint8.
        epsilon : float, probabilidad de tomar una acción aleatoria.
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.array(observation), dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
            q_values = self.online_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def act_greedy(self, observation):
        """Acción puramente greedy (epsilon=0), usada en evaluación."""
        return self.select_action(observation, epsilon=0.0)

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, batch):
        """Un paso de descenso de gradiente sobre un batch muestreado del
        replay buffer.

        Parámetros
        ----------
        batch : tupla (obs, actions, rewards, next_obs, dones) de arreglos
            de NumPy, tal como los retorna ReplayBuffer.sample().

        Retorna
        -------
        loss_value : float
        """
        obs, actions, rewards, next_obs, dones = batch

        obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.uint8, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a) actual, para las acciones que realmente se tomaron.
        q_values = self.online_net(obs_t)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: seleccionar la acción con la red online
            next_q_online = self.online_net(next_obs_t)
            next_actions = torch.argmax(next_q_online, dim=1)

            # evaluando su valor con la red target.
            next_q_target = self.target_net(next_obs_t)
            next_q_sa = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_sa

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradiend clipping. Estabiliza el entrenamiento evitando
        # actualizaciones muy grandes ante recompensas o errores atípicos.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path, map_location=None):
        checkpoint = torch.load(
            path, map_location=map_location or self.device, weights_only=False
        )
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
