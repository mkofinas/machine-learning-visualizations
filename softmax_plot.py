import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        swish = torch.atan(x)
        return swish / swish.sum(1, keepdims=True)


act = Softmax()

t = torch.linspace(-100, 100, 10001)
x = torch.stack([0.9*t, t, 1.1*t], 1)
y = act(x)
Z = torch.diag_embed(y) - y.unsqueeze(2) * y.unsqueeze(1)
dZ = Z[:, [0, 1, 2], [0, 1, 2]]
fig, ax = plt.subplots(3)
ax[0].set_title('Softmax input')
ax[1].set_title('Softmax Output')
ax[0].plot(t, x[:, 0], t, x[:, 1], t, x[:, 2])
ax[1].plot(t, y[:, 0], t, y[:, 1], t, y[:, 2])
ax[2].plot(t, dZ[:, 0], t, dZ[:, 1], t, dZ[:, 2])
ax[2].set_title("Diagonal Gradient")
fig, ax = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        ax[i, j].plot(t, Z[:, i, j])
plt.show()
