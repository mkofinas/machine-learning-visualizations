import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x, y


def gaussian(x, mu=0.0, sigma=1.0):
    return (1 / (sigma * sp.sqrt(2 * sp.pi))) * sp.exp(-0.5 * ((x - mu) / sigma) ** 2)


def logistic(x, x_0=1):
    return 1 / (1 + sp.exp(-x + x_0))


# if __name__ == "__main__":
mu = 6.0
sigma = 1.0
x_0 = 1
N = 50000
EPS = 1e-7

g = sp.log(y) - sp.log(1-y) + 5
g_inv = logistic(x, x_0=5)
p = gaussian(x, mu=mu)
x_v = mu + sigma * np.random.randn(N)
y_v = 1 / (1 + np.exp(-x_v + x_0))

P = gaussian(g, mu=mu)
P2 = P * sp.Abs(sp.diff(g, y))

p1 = sp.plotting.plot_parametric((P, y, (y, EPS, 1-EPS)), show=False)
p2 = sp.plotting.plot_parametric((P2, y, (y, EPS, 1-EPS)), show=False)
p3 = sp.plotting.plot(p, (x, 0, 10), show=False)
p3.append(p1[0])
p3.append(p2[0])
p3.show()

sp.plotting.plot_parametric((P, y, (y, EPS, 1-EPS)))
plt.hist(y_v, bins=30)
