import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')


def plot_reconstruction_errors():
    tau_error = {'mean_error': [],
                 'max_error': [],
                 'min_error': [],
                 'tau': [10.0 ** i for i in range(-18, 0)]}
    for tau in tau_error['tau']:
        dmd.fit(X, svd_rank=1.0 - tau)
        errors = dmd.snapshot_errors
        tau_error['mean_error'].append(np.mean(errors))
        tau_error['max_error'].append(np.max(errors))
        tau_error['min_error'].append(np.min(errors))

    mode_error = {'mean_error': [],
                  'max_error': [],
                  'min_error': [],
                  'n': list(range(1, len(X)))}
    for m in mode_error['n']:
        dmd.fit(X, svd_rank=m)
        errors = dmd.snapshot_errors
        mode_error['mean_error'].append(np.mean(errors))
        mode_error['max_error'].append(np.max(errors))
        mode_error['min_error'].append(np.min(errors))

    from typing import List
    from matplotlib.pyplot import Figure, Axes

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs: List[Axes] = axs.ravel()

    for i, ax in enumerate(axs):
        if i == 0:
            n = mode_error['n']
            ax.set_xlabel(f"# of Modes", fontsize=12)
            ax.set_ylabel(f"Relative $L^2$ Error", fontsize=12)
            ax.semilogy(n, mode_error['mean_error'],
                        '-b*', label="Mean Error")
            ax.semilogy(n, mode_error['max_error'],
                        '-ro', label="Max Error")
            ax.semilogy(n, mode_error['min_error'],
                        '-k+', label="Min Error")
            ax.legend()
            ax.grid(True)
        else:
            tau = tau_error['tau']
            ax.set_xlabel(f"$\\tau$", fontsize=12)
            ax.loglog(tau, tau_error['mean_error'], '-b*', label="Mean Error")
            ax.loglog(tau, tau_error['max_error'], '-ro', label="Max Error")
            ax.loglog(tau, tau_error['min_error'], '-k+', label="Min Error")
            ax.legend()
            ax.grid(True)
    plt.tight_layout()
    plt.show()


def compare_modes():
    from typing import List
    from matplotlib.pyplot import Figure, Axes

    mode_indices = [0, 1, 6 if 6 < len(modes) else len(modes) - 1]
    r = np.array([p.z for p in sim.nodes])
    n_grps = dmd.n_features // len(r)

    for m in mode_indices:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs: List[Axes] = np.ravel(axs)

        for i, ax in enumerate(axs):
            ylabel = r"$\phi_g(r)$" if i == 0 else ""
            if i == 0:
                title = f"$\\alpha$-Eigenfuntion {m}\n" \
                        f"$\\alpha_{m}$ = {alphas[m].real:.3e} " \
                        f"$\\mu$s$^{{-1}}$"
                phi = modes[m].evaluate_mode(r, 0.0, False)
            else:
                title = f"DMD Mode {m}\n" \
                        f"$\omega_{m}$ = {omegas[idx[m]]:.3e} " \
                        f"$\\mu$s$^{{-1}}$"
                phi = dmd.modes[:, idx[m]]

            # Normalize
            argmax = np.argmax(np.abs(phi))
            phi = phi / np.linalg.norm(phi)
            phi *= -1.0 if phi[argmax] < 0.0 else 1.0

            # Plot
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("r (cm)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            for g in range(n_grps):
                ax.plot(r, phi[g::n_grps], label=f"Group {g}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
    plt.show()


########################################
# Get the data
########################################
base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, 'outputs', 'fv')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

X = sim.create_simulation_matrix()
times = sim.times

########################################
# Perform DMD
########################################
from pyROMs import DMD

dmd = DMD(svd_rank=10).fit(X)
dmd.print_summary()

########################################
# Compare to exact eigenvalues
########################################
from modules.neutron_diffusion.analytic import *

exact: AnalyticSolution = load(base + '/sphere6cm.obj')
modes = [exact.get_mode(i, method='amp') for i in range(dmd.n_modes)]
alphas = np.array([mode.alpha for mode in modes])
attn = np.exp(alphas * np.diff(times)[0])

eig_idx = []
for alpha in alphas:
    eig_idx.append(exact.find_mode_index_from_alpha(alpha))
print(eig_idx)

eigs = dmd.eigvals
for i in range(len(eigs)):
    if eigs[i].imag != 0.0:
        omega = np.log(eigs[i]) / (times[1] - times[0])
        if omega.imag % np.pi < 1.0e-12:
            eigs[i] = eigs[i].real + 0.0j

idx = []
omegas = dmd.omegas.real
for i, alpha in enumerate(alphas):
    idx.append(np.argmin(np.abs(alpha - omegas)))

plt.figure()
plt.xlabel(r"$\mathcal{R}~(\lambda)$", fontsize=14)
plt.ylabel(r"$\mathcal{I}~(\lambda)$", fontsize=14)
plt.plot(eigs.real, eigs.imag, 'bo', label='DMD')
plt.plot(attn.real, attn.imag, 'r*', label='Exact')
plt.grid(True)
plt.legend()
plt.tight_layout()

compare_modes()

msg = "\\begin{tabular}{|c|c|c|}\n\t\hline" \
      "\n\t\\textbf{Analytic Eigenvalue} & " \
      "\\textbf{DMD Eigenvalue} & " \
      "\\textbf{Relative Difference} \\\\\hline"
for m in range(dmd.n_modes):
    alpha, omega = alphas[m].real, omegas[idx[m]]
    msg += f"\n\t\hline {alpha:.3e} & " \
           f"{omega:.3e} & " \
           f"{(alpha - omega) / np.abs(alpha):.3e} \\\\"
msg += "\n\t\hline\n\end{tabular}"
print(msg)
plt.show()
