import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')


def plot_reconstruction_errors():
    tau_error = {'mean_error': [],
                 'max_error': [],
                 'min_error': [],
                 'total_error': [],
                 'tau': [10.0 ** i for i in range(-16, 0)]}
    for tau in tau_error['tau']:
        dmd.fit(X, svd_rank=tau, opt=False)
        errors = dmd.snapshot_errors

        tau_error['mean_error'].append(np.mean(errors))
        tau_error['max_error'].append(np.max(errors))
        tau_error['min_error'].append(np.min(errors))
        tau_error['total_error'].append(dmd.reconstruction_error)

    mode_error = {'mean_error': [],
                  'max_error': [],
                  'min_error': [],
                  'total_error': [],
                  'n': list(range(1, len(X)))}
    for m in mode_error['n']:
        dmd.fit(X, svd_rank=m)
        errors = dmd.snapshot_errors
        mode_error['mean_error'].append(np.mean(errors))
        mode_error['max_error'].append(np.max(errors))
        mode_error['min_error'].append(np.min(errors))
        mode_error['total_error'].append(dmd.reconstruction_error)

    from typing import List
    from matplotlib.pyplot import Figure, Axes

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs: List[Axes] = axs.ravel()

    for i, ax in enumerate(axs):
        if i == 0:
            n = mode_error['n']
            ax.set_xlabel(f"# of Modes", fontsize=12)
            ax.set_ylabel(f"Relative $L^2$ Error", fontsize=12)
            ax.semilogy(n, mode_error['mean_error'], '-*b',
                        ms=4.0, label="Mean Error")
            ax.semilogy(n, mode_error['max_error'], '-or',
                        ms=4.0, label="Max Error")
            ax.semilogy(n, mode_error['min_error'], '-+k',
                        ms=4.0, label="Min Error")
            # ax.semilogy(n, mode_error['total_error'], '-^g',
            #             ms=3.0, label="Total Error")
        else:
            tau = tau_error['tau']
            ax.set_xlabel(r"$\tau_{cut}$", fontsize=12)
            ax.loglog(tau, tau_error['mean_error'], '-*b',
                      ms=4.0, label="Mean Error")
            ax.loglog(tau, tau_error['max_error'], '-or',
                      ms=4.0, label="Max Error")
            ax.loglog(tau, tau_error['min_error'], '-+k',
                      ms=4.0, label="Min Error")
            # ax.loglog(tau, tau_error['total_error'], '-^g',
            #           ms=3.0, label="Total Error")
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.tick_params(labelsize=12)
    plt.tight_layout()


def compare_modes(mode=0):
    from typing import List
    from matplotlib.pyplot import Figure, Axes

    n_grps = dmd.n_features // len(r)

    if mode == 0:
        mode_indices = [0, 1, 2, -2, -1]
    else:
        argmax_eig = np.argmax(dmd.eigvals.real)
        argmax_b = np.argmax(np.abs(dmd.amplitudes.real))
        mode_indices = [argmax_eig, argmax_b]

    for m in mode_indices:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs: List[Axes] = np.ravel(axs)

        m = m if m >= 0 else dmd.n_modes + m
        m_dmd = idx[m] if mode == 0 else m
        for i, ax in enumerate(axs):
            ylabel = r"$\phi_g(r)$" if i == 0 else ""
            if i == 0:
                title = f"$\\alpha$-Eigenfuntion {m}\n" \
                        f"$\\alpha_{m}$ = {alphas[m].real:.3e} " \
                        f"$\\mu$s$^{{-1}}$"
                phi = modes[m].evaluate_mode(r, 0.0, False)
            else:
                title = f"DMD Mode {m}\n" \
                        f"$\omega_{m}$ = {omegas[m_dmd]:.3e} " \
                        f"$\\mu$s$^{{-1}}$"
                phi = dmd.modes[:, m_dmd]

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
            ax.legend(fontsize=12)
            ax.grid(True)
            ax.tick_params(labelsize=12)
        plt.tight_layout()


########################################
# Get the data
########################################
base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, 'outputs', 'fv')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

X = sim.create_simulation_matrix()
r = np.array([p.z for p in sim.nodes])
times = sim.times

########################################
# Perform DMD
########################################
from pyROMs import DMD

dmd = DMD(svd_rank=0, opt=False)
plot_reconstruction_errors()

dt = np.diff(times)[0]
dmd.original_time = {'t0': 0.0, 'tend': times[-1], 'dt': dt}
dmd.fit(X, svd_rank=1.0e-8)
dmd.print_summary(skip_line=True)
# dmd.plot_singular_values()

# errors = dmd.snapshot_errors
# plt.figure()
# plt.tick_params(labelsize=12)
# plt.xlabel("Time ($\mu$s)", fontsize=12)
# plt.ylabel("Relative $L^2$ Error", fontsize=12)
# plt.semilogy(times, errors, '-*b', ms=4.0)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

count = 0
for m, mode in enumerate(dmd.modes.T):
    if np.sum(np.abs(mode[2::3])) > dmd.reconstruction_error:
        count += 1
print(f"\nModes with thermals:\t{count}")


########################################
# Compare to exact eigenvalues
########################################
from modules.neutron_diffusion.analytic import *

filename = f"{base}/sphere6cm_thermal_1mb"
exact: AnalyticSolution = load(f"{filename}.obj")
modes = [exact.get_mode(i, method='amp') for i in range(dmd.n_modes)]
alphas = np.array([mode.alpha for mode in modes])
attn = np.exp(alphas * np.diff(times)[0])

eig_idx = []
for alpha in alphas:
    eig_idx.append(exact.find_mode_index_from_alpha(alpha))
print()
print(eig_idx)

ind = np.argsort(np.abs(dmd.amplitudes.real))[::-1]
eigs = dmd.eigvals.real[ind]
print()
print(f"Thermal Index:\t{np.argmax(eigs)}")

eigs = dmd.eigvals
for i in range(len(eigs)):
    if eigs[i].imag != 0.0:
        omega = np.log(eigs[i]) / (times[1] - times[0])
        if omega.imag % np.pi < 1.0e-12:
            eigs[i] = eigs[i].real + 0.0j

omegas = dmd.omegas.real
idx, avail = [], list(range(len(omegas)))
for i, alpha in enumerate(alphas):
    idx.append(avail[np.argmin(np.abs(alpha - omegas[avail]))])
    avail.remove(idx[i])

plt.figure()
plt.xlabel(r"$\mathcal{R}~(\lambda)$", fontsize=14)
plt.ylabel(r"$\mathcal{I}~(\lambda)$", fontsize=14)
plt.plot(eigs.real, eigs.imag, 'bo', label='DMD')
plt.plot(attn.real, attn.imag, 'r*', label='Exact')
plt.grid(True)
plt.legend(fontsize=12)
plt.tick_params(labelsize=12)
plt.tight_layout()

compare_modes(1)

alpha_fund = exact.get_mode(0, method='eig').alpha.real
omega_fund = max(omegas)
print()
print(f"Analytic:\t{alpha_fund:.5e}")
print(f"DMD:\t\t{omega_fund:.5e}")
print(f"Diff:\t\t{1.0-omega_fund/alpha_fund:.3e}")
print()

msg = "\\begin{tabular}{|c|c|c|}\n\t\hline" \
      "\n\t\\textbf{Analytic Eigenvalue} & " \
      "\\textbf{DMD Eigenvalue} & " \
      "\\textbf{Relative Difference} \\\\\hline"
for m in range(dmd.n_modes):
    alpha, omega = alphas[m].real, omegas[idx[m]]
    msg += f"\n\t\hline {alpha:.5e} & " \
           f"{omega:.5e} & " \
           f"{(alpha - omega) / np.abs(alpha):.3e} \\\\"
msg += "\n\t\hline\n\end{tabular}"
print()
print(msg)
plt.show()
