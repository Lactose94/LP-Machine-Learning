import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib

def float_to_str(nr: float):
    nr = float(nr)
    return str(nr).replace('.', '')

def analyze(n: int, min_sigma, max_sigma, modi, stepsize, max_n=-1):
    fit_path = f'fit_{n}_({float_to_str(min_sigma)}-{float_to_str(max_sigma)})_m{modi}_s{stepsize}.dat'
    fit = np.loadtxt(fit_path)

    pred_path = f'prediction_{n}_({float_to_str(min_sigma)}-{float_to_str(max_sigma)})_m{modi}_s{stepsize}.dat'
    prediction = np.loadtxt(pred_path)

    plt.scatter(fit[:, 0], fit[:, 1])
    plt.show()

nqs = [8, 10, 12, 14]

legends = []
fig, axs = plt.subplots(1, 2, figsize = (16, 6))
titles = ["$\Delta F^ *$", "$\Delta E^*$"]
axs[0].grid()
#axs[0].set_yscale('log')
axs[1].grid()
axs[1].set(xlabel="$\sigma$", title=titles[1])

for modi in nqs:
    fit_path = f'fit_{20}_({float_to_str(0.0)}-{float_to_str(10.0)})_m{modi}_s{100}.dat'

    fit = np.loadtxt(fit_path)

    pred_path = f'prediction_{20}_({float_to_str(0.0)}-{float_to_str(10.0)})_m{modi}_s{100}.dat'
    prediction = np.loadtxt(pred_path)

    axs[0].scatter(fit[0:, 0], fit[0:, 1], s=20)
    axs[0].set(xlabel="$\sigma$", title=titles[0])
    axs[1].scatter(fit[:, 0], fit[:, -1], s=20)
    legends.append(f'$N_q=${modi}')

axs[0].legend(legends)
axs[1].legend(legends)

plt.show()