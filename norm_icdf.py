import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


p = stats.norm()
x = np.linspace(-5, 5, 1001)

pp = np.linspace(0.05, 0.95, 19)
pp0 = np.vstack([pp, np.zeros_like(pp)])
pp1 = np.vstack([p.ppf(pp), np.ones_like(pp)])

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(x, p.pdf(x))
ax[0].set_title(r'Gaussian PDF $N(0, 1)$')
ax[1].plot(x, p.cdf(x))
ax[0].set_title(r'Gaussian CDF $N(0, 1)$')
ax[2].scatter(pp0[0], pp0[1], color='orange')
ax[2].scatter(pp1[0], pp1[1], color='purple')
for pi0, pi1 in zip(pp0.T, pp1.T):
    ax[2].plot([pi0[0], pi1[0]], [pi0[1], pi1[1]], 'k')

for i in range(len(pp)-1):
    px = np.linspace(p.ppf(pp[i]), p.ppf(pp[i+1]), 100)
    ax[0].fill_between(px, p.pdf(px), alpha=0.2)

ax[1].scatter(pp1[0], p.cdf(pp1[0]))
for pp1i in pp1[0]:
    ax[1].plot([-5, pp1i], [p.cdf(pp1i), p.cdf(pp1i)], 'k', linestyle='--',
               linewidth=1, alpha=0.5)

ax[3].plot(x, p.ppf(x))
plt.show()
