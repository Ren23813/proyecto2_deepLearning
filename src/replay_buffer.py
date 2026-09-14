"""
replay_buffer.py
----------------
Buffer de repetición de experiencias (experience replay) para DQN.

Se implementa con arreglos de NumPy pre-alocados (en vez de una lista de
Python de tuplas) para que el buffer sea eficiente en memoria: cada
observación (4, 84, 84) se guarda en uint8, no en float32, lo que reduce el
uso de RAM. Esto es importante porque si no se llena la memoria muy rápido, 
haciendo imposible el entrenamiento
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape=(4, 84, 84)):
        self.capacity = capacity
        self.obs_shape = obs_shape

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng=None):
        rng = rng or np.random
        idx = rng.randint(0, self.size, size=batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
