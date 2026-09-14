"""
model.py
--------
Arquitectura de red neuronal para Dueling Double DQN sobre observaciones de
Atari preprocesadas (4, 84, 84), siguiendo la arquitectura convolucional del
paper original de DQN (Mnih et al., 2015) con la modificación "dueling".

Arquitectura
------------
Entrada: tensor (batch, 4, 84, 84), uint8 en [0, 255].
1. Normalización: dividir por 255.0 -> float32 en [0, 1].
2. Conv2d(4  -> 32, kernel=8, stride=4) + ReLU   -> (32, 20, 20)
3. Conv2d(32 -> 64, kernel=4, stride=2) + ReLU   -> (64, 9, 9)
4. Conv2d(64 -> 64, kernel=3, stride=1) + ReLU   -> (64, 7, 7)
5. Flatten -> 64*7*7 = 3136
6. FC(3136 -> 512) + ReLU (tronco compartido)
7. Dos cabezas ("dueling"):
   - Value stream:     FC(512 -> 512) + ReLU -> FC(512 -> 1)
   - Advantage stream: FC(512 -> 512) + ReLU -> FC(512 -> n_actions)
8. Combinación:
   Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
   La resta de la media (en vez de solo sumar V + A) es la que le da
   identificabilidad al descomponer Q en V y A

Esta arquitectura se usa tanto para la red "online" (la que se entrena y
decide acciones) como para la red "target" (usada para calcular el valor
objetivo en Double DQN), ambas con la misma clase pero pesos independientes.
"""

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        super().__init__()
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        conv_out_size = self._get_conv_out((in_channels, 84, 84))

        self.fc_tronco = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(torch.flatten(o, 1).shape[1])

    def forward(self, x):
        # x: (batch, 4, 84, 84), uint8 o float
        x = x.float() / 255.0
        conv_out = self.conv(x)
        conv_out = torch.flatten(conv_out, start_dim=1)
        tronco = self.fc_tronco(conv_out)

        value = self.value_stream(tronco)                  # (batch, 1)
        advantage = self.advantage_stream(tronco)           # (batch, n_actions)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    net = DuelingDQN(in_channels=4, n_actions=6)
    x = torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)
    q = net(x)
    print("Q shape:", q.shape)  # esperado: (8, 6)
    n_params = sum(p.numel() for p in net.parameters())
    print("Número de parámetros:", n_params)
