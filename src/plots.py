import matplotlib.pyplot as plt
import numpy as np


# def print_timescales(eigvals):
#     """Simulation time = 3 * 250 ns = 750,000 ps
#     Number of frames = data.shape[0] = 750,000
#     Therefore, tau = 1 ps
#     """
#     from .vampnet import timescale

#     ferguson_srv_ts = (1627, 72, 33, '-', '-', '-')
#     tau = 1
#     for i, (t, tf) in enumerate(zip(timescale(tau, eigvals), ferguson_srv_ts)):
#         print(f'tau_{i+1} = {t:.1f} ps  ({tf} ps)')


def check_eigfuncs(psi):
    # proper expecation value, norm
    for i in range(psi.shape[1]):
        print(f"E[psi_{i+1}] = {np.mean(psi[:, i]):.2e} \t E[psi_{i+1}^2] = {np.mean(psi[:, i] ** 2):.5f}")
    # orthogonality
    for i in range(psi.shape[1]):
        for j in range(i + 1, psi.shape[1]):
            print(f"E[psi_{i+1} psi_{j+1}] = {np.mean(psi[:, i] * psi[:, j]):.2e}")


def plot_dihedrals_hist2d(fig, ax, dihedrals):
    h = ax.hist2d(*dihedrals.T, bins=75, density=True, cmin=1e-10, cmap='magma_r')
    fig.colorbar(h[3], ax=ax)
    ax.patch.set_facecolor('0.6')
    ax.set_xlabel(r'$\phi$', fontsize=12)
    ax.set_ylabel(r'$\psi$', fontsize=12, rotation=0)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])


def plot_eigfuncs(fig, axs, theta, psi_grid, timescales):
    for i, ax in enumerate(axs.flatten()):
        if i % axs.shape[1] == 0: # left
            ax.set_ylabel(r'$\psi$', fontsize=12, rotation=0)
        if i >= np.prod(axs.shape) - axs.shape[1]: # bottom
            ax.set_xlabel(r'$\phi$', fontsize=12)
        cb = ax.pcolormesh(theta, theta, psi_grid[:, :, i].T)
        fig.colorbar(cb, ax=ax)
        ax.text(1.8, 2.2, f'$\psi_{i+1}$', fontsize=18)
        timescale = timescales[i]
        if timescale >= 10_000:
            timescale /= 1000
            units = 'ns'
        else:
            units = 'ps'
        ax.text(1.4, 1.2, f'{timescale:.1f} {units}')


def plot_cvs_hist2d(fig, ax, s1, s2, cvs):
    h = ax.hist2d(*cvs.T, bins=[s1, s2], density=True, cmin=1e-6, cmap='magma_r')
    fig.colorbar(h[3], ax=ax);
    ax.patch.set_facecolor('0.6')
    ax.set_xlabel("$s_1$", fontsize=14)
    ax.set_ylabel("$s_2$", fontsize=14, rotation=0)


def plot_traj_energy(ax, fn_out, width, cutoff):
    """width : number of points for moving average
    cutoff : max value for energies
    """
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    outdata = np.genfromtxt(fn_out, delimiter=',')
    print(f'{outdata.shape = }')
    pe = outdata.T[0]; pe[pe > cutoff] = cutoff
    ke = outdata.T[1]; ke[ke > cutoff] = cutoff
    ax.plot(pe, alpha=0.6, label='Potential Energy')
    ax.plot(moving_average(pe, width), 'k-', lw=1, alpha=0.4)
    ax.plot(ke, alpha=0.6, label='Kinetic Energy')
    ax.plot(moving_average(ke, width), 'k-', lw=1, alpha=0.4)
    ax.legend()


def plot_state_labels(axs, sd, dill_states_grid, psi_states_grid):
    cb = axs[0].pcolormesh(sd.theta_grid, sd.theta_grid, dill_states_grid.T, alpha=0.8)
    cb = axs[1].pcolormesh(sd.theta_grid, sd.theta_grid, psi_states_grid.T, alpha=0.8)
    for ax in axs: # [1:]:
        ax.text(-2.7, 2.3, r'$C_5$', color='white', fontsize=16)
        ax.text(-1.5, 2.3, r'$C_7^{eq}$', color='white', fontsize=16)
        ax.text(0.8, 2.3, r'$C_7^{ax}$', color='black', fontsize=16)
        ax.text(-2.9, 0, r'$\alpha_P$', color='black', fontsize=16)
        ax.text(-1.7, 0, r'$\alpha_R$', color='black', fontsize=16)
        ax.text(0.8, 0, r'$\alpha_L$', color='black', fontsize=16)
    for ax in axs:
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\psi$', rotation=0)


def plot_trajectory(axs, sd):
    tau_ns = sd.lagtime / sd.lagframes / 1000
    time_ns = np.arange(0, len(sd.dihedrals) * tau_ns, tau_ns)
    states = np.load(f'{sd.dir}/states_psi.npy')

    # wrap C7eq states together
    phi_t = sd.dihedrals[:, 0].copy()
    phi_t[phi_t > 2] -= 2 * np.pi

    axs[0].plot(time_ns, phi_t, '-', linewidth=0.1)
    axs[0].set_ylabel(r'$\phi$', rotation=0, fontsize=12)
    axs[0].set_xticklabels('')

    axs[1].plot(time_ns, sd.cvs[:, 0], '-', linewidth=0.4)
    axs[1].set_ylabel(r'$\psi_1$', rotation=0, fontsize=12)
    axs[1].set_xticklabels('')

    state_labels = [r'$C_5$', r'$C_7^{eq}$', r'$\alpha_P$', r'$\alpha_R$', r'$C_7^{ax}$', r'$\alpha_L$']
    axs[2].plot(time_ns, states, '-', linewidth=0.1, markersize=0.4)
    axs[2].set_xlabel('Time (ns)')
    axs[2].set_yticks(list(range(1, 7)))
    axs[2].set_yticklabels(state_labels, rotation=0, fontsize=12);


def plot_chapman_kolmogorov(ax, kvals, ck_srv, ck_psi, ck_dill):
    ax.plot(kvals, ck_srv, '-', linewidth=3, alpha=0.4, color='black', label='Deep TICA')
    ax.plot(kvals, ck_psi, '-', linewidth=3, alpha=0.6, label='MSM, eig')
    ax.plot(kvals, ck_dill, '-', linewidth=3, alpha=0.6, label='MSM, grid')
    ax.set_xlabel(r'$k$', fontsize=12)
    # ax.set_ylabel(r'MSE($~T(k\tau),~T(\tau)^k~$)', fontsize=12)
    ax.set_ylabel(r'$\Vert T(k\tau)-T(\tau)^k\Vert_F$', fontsize=12)
    ax.legend(loc='upper left')


def plot_timescales(ax, kvals, sd, timescales_srv, timescales_psi, timescales_dill):
    for i, (ts_srv, ts_psi, ts_dill) in enumerate(zip(timescales_srv.T,
                                                      timescales_psi.T,
                                                      timescales_dill.T)):
        ktau = kvals * sd.lagtime
        if i == 0:
            lb_srv, lb_psi, lb_dill = 'Deep TICA','MSM, eig', 'MSM, grid'
        else:
            lb_srv = lb_psi = lb_dill = ''
        ax.semilogy(ktau, ts_srv, linewidth=2, alpha=0.6, label=lb_srv, color='black')
        ax.semilogy(ktau, ts_psi, linewidth=2, alpha=0.6, label=lb_psi, color='C0')
        # ax.semilogy(k, ts_dill, linewidth=3, alpha=0.6, label=lb_dill, color='C1')
        plt.fill_between(ktau, ktau, color= 'blue', alpha= 0.02)
    ax.text(5, 1000, r'$\tau_1$', fontsize=12)
    ax.text(5, 100, r'$\tau_2$', fontsize=12)
    ax.text(5, 15, r'$\tau_3$', fontsize=12)
    ax.text(5, 4, r'$\tau_4$', fontsize=12)
    ax.text(5, 1.1, r'$\tau_5$', fontsize=12)
    ax.set_xlabel(r'$k\tau$ (ps)', fontsize=12)
    ax.set_ylabel(r'Timescale (ps)', fontsize=12)
    ax.legend(loc=(0.6, 0.7))


def plot_eigvals(ax, sd, eigvals_psi, eigvals_dill):
    ax.bar(np.arange(1, 6) + -0.25, sd.eigvals[:-1], color='k', width = 0.25, alpha=0.4, label='Deep TICA')
    ax.bar(np.arange(1, 6) + 0, eigvals_psi[0], color = 'C0', width = 0.25, alpha=0.4, label='MSM, eig')
    ax.bar(np.arange(1, 6) + 0.25, eigvals_dill[0], color = 'C1', width = 0.25, alpha=0.4, label='MSM, grid')
    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$\tilde{\lambda}_k$', rotation=0, fontsize=14)
    ax.yaxis.set_label_coords(-0.12, 0.55)
    ax.text(1, 0.3, '1442 ps', ha='center', fontsize=14)
    ax.text(2, 0.3, '60 ps', ha='center', fontsize=14)
    ax.text(3, 0.3, '27 ps', ha='center', fontsize=14)
    ax.text(4, 0.3, '2.3 ps', ha='center', fontsize=14)
    ax.text(5, 0.3, '1.5 ps', ha='center', fontsize=14)
    ax.legend()
















def plot_bias_dihedral_cvs(fig, ax1, ax2, theta, bias_dihedral,
                           s1, s2, bias_cvs):
    cb = ax1.pcolormesh(theta, theta, bias_dihedral.T)
    ax1.set_xlabel(r'$\phi$', fontsize=14)
    ax1.set_ylabel(r'$\psi$', fontsize=14, rotation=0)
    ax1.patch.set_facecolor('0.6')

    cb = ax2.pcolormesh(s1, s2, bias_cvs.T)
    ax2.set_xlabel("$s_1$", fontsize=14)
    ax2.set_ylabel("$s_2$", fontsize=14, rotation=0)
    fig.colorbar(cb, ax=ax2)
    fig.tight_layout(pad=2.0)
