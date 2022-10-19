import numpy as np
import torch
import openmm.unit as unit
import mdtraj as md
from dataclasses import dataclass, field
import _pickle as cPickle
import os


def to_torch(a, device=None):
    """Send array to torch.Tensor.  If device=None, it will go to cpu (always?)  The intention is for a single utility function to handle this everywhere.

    torch.tensor():
        Takes array-like (np.ndarrays, lists, ...)
        Makes a copy of the data unless they reside on same device and have
        corresponding data types.
    """
    return torch.tensor(a, dtype=torch.float32, device=device)
#     return torch.Tensor(a)
#     return torch.as_tensor(a, dtype=torch.float32)
#     return torch.from_numpy(a)
#     return torch.from_numpy(np.asarray(a, dtype=np.float32))


def save_npz(filename, obj):
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f)


def load_npz(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)


# class SimulationParameters:
#     def __init__(self, simulation_time,
#                  report_interval_time,
#                  num_simulations):

#         self.temperature = 300 * unit.kelvin
#         self.timestep = 0.002 * unit.picosecond
#         self.friction_coeff = 1 / unit.picosecond

#         self.tau_G = 120 * unit.femtoseconds
#         # self.tau_G = 1 * unit.picosecond
#         self.height = 1.20 * unit.kilojoule_per_mole
#         # self.height = 0.418 * unit.kilojoule_per_mole
#         self.bias_factor = 5
#         self.width = np.array([0.1, 0.1])

#         self.simulation_time = simulation_time
#         self.ns = f'{simulation_time.in_units_of(unit.nanosecond)._value:.1f}'
#         self.report_interval = round(report_interval_time / self.timestep)
#         self.num_gaussians = round(simulation_time / self.tau_G)
#         self.steps_per_gaussian = round(self.tau_G / self.timestep)

#         self.num_simulations = num_simulations
#         self.total_simulation_time = num_simulations * simulation_time
#         self.total_ns = f'{self.total_simulation_time.in_units_of(unit.nanosecond)._value:.1f}'


def pretty_print(obj):
    for field in obj.__dataclass_fields__:
        print(f'{field} = {getattr(obj, field)}')


@dataclass
class SimulationParameters:
    simulation_time: unit.quantity.Quantity
    report_interval_time: unit.quantity.Quantity
    num_simulations: int

    temperature: unit.quantity.Quantity = 300 * unit.kelvin
    timestep: unit.quantity.Quantity = 0.002 * unit.picosecond
    friction_coeff: unit.quantity.Quantity = 1 / unit.picosecond

    report_interval: int = field(init=False)
    total_simulation_time: unit.quantity.Quantity = field(init=False)
    ns: str = field(init=False)
    total_ns: str = field(init=False)

    def __post_init__(self):
        self.report_interval = round(self.report_interval_time / self.timestep)
        self.total_simulation_time = self.num_simulations * self.simulation_time
        self.ns = f'{self.simulation_time.in_units_of(unit.nanosecond)._value:.1f}'
        self.total_ns = f'{self.total_simulation_time.in_units_of(unit.nanosecond)._value:.1f}'

@dataclass
class SimulationParametersMetaD(SimulationParameters):
    tau_G: unit.quantity.Quantity = 120 * unit.femtoseconds
    height: unit.quantity.Quantity = 1.20 * unit.kilojoule_per_mole
    width: np.ndarray = np.array([0.1, 0.1])
    bias_factor: float = 5.0

    num_gaussians: int = field(init=False)
    steps_per_gaussian: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.num_gaussians = round(self.simulation_time / self.tau_G)
        self.steps_per_gaussian = round(self.tau_G / self.timestep)


def write_pdb(simulation, filename):
    from openmm.app import PDBFile
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(filename, 'w'))


def create_system(forcefield, topology):
    from openmm.app import HBonds, PME
    return forcefield.createSystem(topology,
                                   nonbondedMethod=PME,
                                   nonbondedCutoff=0.9 * unit.nanometer,
                                   constraints=HBonds)


def state_data_reporter(filename, report_interval):
    from openmm.app import StateDataReporter
    return StateDataReporter(filename,
                             report_interval,
                             kineticEnergy=True,
                             potentialEnergy=True,
                             temperature=True)


def hdf5_reporter(filename, report_interval):
    from mdtraj.reporters import HDF5Reporter
    return HDF5Reporter(filename, report_interval)


class SimulationCVS:
    """Used to compute CV's from
        (a) pdb files
        (b) simulation.context.getState

    This is required by Metadynamics.add_gaussian(cvs)
    """
    def __init__(self, pdb_filename):
        self.pdb_filename = pdb_filename

        traj = md.load(pdb_filename)
        self.ala_atoms = traj.topology.select('resname != HOH')

        heavy_atoms = traj.topology.select('resname != HOH && type != H')
        self.atom_pairs = np.array([(i, j)
                                    for i in heavy_atoms
                                    for j in heavy_atoms if i < j])

    def pdb(self):
        from openmm.app import PDBFile
        return PDBFile(self.pdb_filename)

    def pdb_cvs(self, net):
        traj = md.load(self.pdb_filename)
        xyz = traj[0].xyz.squeeze()
        return self.xyz_to_cvs(net, xyz)

    def sim_cvs(self, net, simulation):
        state = simulation.context.getState(getPositions=True)
        xyz = np.array(state.getPositions()._value)
        return self.xyz_to_cvs(net, xyz)

    def xyz_to_cvs(self, net, xyz):
        positions = to_torch(xyz[self.ala_atoms])
        x = self.featurize(positions)
        return net(x).numpy().flatten()

    def featurize(self, positions):
        return torch.stack([
            torch.sqrt(torch.sum(torch.square(positions[j] - positions[i])))
            for (i, j) in self.atom_pairs
        ]).unsqueeze(0)


# def pdb_to_numpy_cvs(filename, net):
#     def featurize(positions, atom_pairs):
#         return torch.stack([
#             torch.sqrt(torch.sum(torch.square(positions[j] - positions[i])))
#             for (i, j) in atom_pairs
#         ]).unsqueeze(0)

#     traj = md.load(filename)
#     ala_atoms = traj.topology.select('resname != HOH')
#     heavy_atoms = traj.topology.select('resname != HOH && type != H')
#     atom_pairs = np.array([(i, j)
#                            for i in heavy_atoms
#                            for j in heavy_atoms if i < j])

#     xyz = traj[0].xyz.squeeze()
#     positions = to_torch(xyz[ala_atoms])
#     x = featurize(positions, atom_pairs)
#     return net(x).numpy().flatten()


def timescale_from_acf(st, lagtime, num_points):
    """Do timescale analysis using CV trajectory s(t)

    Parameters
    ----------
    st : ndarray with shape (num_timesteps,)
        Full trajectory of CV
    lagtime : float
        Time separating trajectory frames.  Units will match timescale estimate
    num_points : int
        How far out to compute ACF (this is expensive).
        Timescale = int_0^infty dt C(t)

    Return
    ------
    tau : float
        Computed timescale for CV
    ct : ndarray with shape (num_points,)
        The ACF, C(t)
    """
    def acf(st, timestep):
        if timestep == 0:
            return np.mean(st * st)
        else:
            return np.mean(st[:-timestep] * st[timestep:])

    c0 = acf(st, 0)
    ct = np.array([acf(st, t) for t in range(num_points)]) / c0
    tau = lagtime * ct.sum()
    return tau, ct


def dill_state(phi_, psi_):
    def rad2deg(theta):
        return theta * 180 / np.pi

    phi_, psi_ = rad2deg(phi_), rad2deg(psi_)
    if phi_ < -105 or phi_ > 117: # 1, 3
        if -124 < psi_ < 28:
            return 3
        else:
            return 1
    elif -105 < phi_ < 0: # 2, 4
        if -124 < psi_ < 28:
            return 4
        else:
            return 2
    else: # 5, 6
        if -5 < psi_ < 111:
            return 6
        else:
            return 5


def psi_state(psi):
    if np.isnan(psi[0]):
        return np.nan
    if psi[0] > -0.5: # 1,2,3,4
        if psi[1] > 0: # 1,2
            if psi[3] > 0.5:
                return 2
            else:
                return 1
        else:
            if psi[4] > 2: # 3,4
                return 3
            else:
                return 4
    else: # 5,6
        if psi[2] > -0.5:
            return 6
        else:
            return 5


def transition_counts_matrix(states, data_dir, label, lagframes=1,
                             recalculate=False):
    """Count transitions in state data separated by 'lagframes' frames.  If saved results are found in a file and recalculate=False, just read file and return that instead.

    Parameters
    ----------
    states : int ndarray, shape (num_frames,)
        Assigned state numbers in [0, 5]
    data_dir : str
        Location of saved data files
    label : str
        Distinguish between 'dill' and 'psi'
    lagframes : int
        Number of frames separating transitions
    recalculate : bool
        Flag in case you want to recalculate when the data file exists

    Return
    ------
    counts : int ndarray, shape (6, 6)
        Matrix containing the counts of the transitions found in the state data
    """
    fn = f'{data_dir}/transition_counts_{label}_lag={lagframes}.npy'
    if not os.path.isfile(fn) or recalculate:
        counts = np.full((6, 6), 0, dtype=int)
        for s0, s1 in zip(states[:-lagframes], states[lagframes:]):
            counts[(s0 - 1), (s1 - 1)] += 1
        np.save(fn, counts)
        return counts
    else:
        return np.load(fn)


def pi_mle(counts):
    """MLE estimate for the equilibrium distribution from transition counts.
            pi_i ~ sum_j C_ij / sum_ij C_ij
    """
    counts_sym = 0.5 * (counts + counts.T)
    return counts_sym.sum(axis=1) / counts_sym.sum()


def transition_matrix_mle(counts):
    """MLE estimate for the transition matrix from transition counts.
            T_ij ~ C_ij / sum_j C_ij
    """
    counts_sym = 0.5 * (counts + counts.T)
    return counts_sym / counts_sym.sum(axis=1).reshape(-1, 1)


def chi_data(states):
    """Convert state label array to one-hot chi matrix.

    Parameters:
    ----------
    states : int ndarray, shape (num_frames,)
        State labels, integers in [1, num_states]

    Return:
    ------
    chis : int ndarray, shape (num_states, num_frames)
        Full trajectory of one-hot columns, chi
    """
    assert states.min() == 1, 'state labels must start at 1'

    chis = np.zeros((states.max(), len(states)), dtype=int)
    for state, chi in zip(states, chis.T):
        chi[state - 1] = 1
    return chis


def koopman_matrix_dmd(state_data, lagframes=1):
    """Parameters:
    ----------
    state_data : ndarray, shape (num_states, num_frames)
        Column-wise state representations [x_0 .. x_{T-1}]

    Return:
    ------
    K(k tau) : ndarray, shape (num_states, num_states)
        DMD estimate for Koopman matrix
    """
    X = state_data[:, :-lagframes]
    Y = state_data[:, lagframes:]
    return np.linalg.inv(X @ X.T) @ (X @ Y.T)


# def chapman_kolmogorov(tmatrices):
#     """Do Chapman-Kolmogorov test on a list of transition matrices:
#         [T(tau), ..., T(n tau)]

#     Return : list of terms MSE( T(k * tau), T(tau)^k )
#     """
#     def mse(a, b):
#         return ((a - b)**2).mean()

#     return [mse(tm, np.linalg.matrix_power(tmatrices[0], k + 1))
#             for k, tm in enumerate(tmatrices)]


def eig_sorted(a):
    """
    Return : w = (n,) eigenvalues, sorted largest to smallest
             v = (n, n) right eigenvectors, sorted
    """
    w, v = np.linalg.eig(a)
    idx = w.argsort()[::-1]
    return np.real(w[idx]), np.real(v[:, idx])


def eig_transition_matrix(T):
    """Eigenvectors normalized per Noe:
        <phi_i|psi_j> = <phi_i|phi_j>_pi^-1 = <psi_i|psi_j>_pi = delta_ij

    Parameters
    ----------
    T : ndarray, size (n, n)
        Transition matrix

    Return
    ------
    w : ndarray, size (n,)
        Eigenvalues
    vl : ndarray, size (n, n)
        Left eigenvectors, row-wise (phi)
    vr : ndarray, size (n, n)
        Right eigenvectors, col-wise (psi)
    """
    w, vl = eig_sorted(T.T)
    vl = vl.T
    vr = np.zeros(vl.shape)
    pi = vl[0, :] / vl[0, :].sum()

    for i, phi in enumerate(vl):
        if i == 0:
            vl[i, :] = pi
        else:
            vl[i, :] = phi / np.sqrt(np.inner(phi, phi / pi))
        vr[:, i] = vl[i, :] / pi
    return w, vl, vr


# def eig_koopman_msm(counts):
#     """Solve the eigenvalue problem the usual Koopman's way for an MSM where:

#         C0 = diag(pi) = E[chi(x_t) chi(x_t)']
#         C1 = P = E[chi(x_t) chi(x_{t+\tau})']

#         K = C0^{-1/2} C1 C0^{-1/2}  =>  K Q = Q Lambda
#         A = C0^{-1/2} Q

#         tilde{\psi}_i(x_t) = A_ki if x_t in s_k

#         Return:
#         -------
#         w : ndarray with shape (n,)
#             Eigenvalues
#         A : ndarray with shape (n, n)
#             The expansion coefficients of the eigenvectors in terms of {chi_i(x)}
#     """
#     counts_sym = 0.5 * (counts + counts.T)
#     P = counts_sym / counts_sym.sum()
#     pi = pi_mle(counts)

#     K = np.diag(1 / np.sqrt(pi)) @ P @ np.diag(1 / np.sqrt(pi))
#     w, v = eig_sorted(K)
#     return w, np.diag(1 / np.sqrt(pi)) @ v


def eigvals(tmatrix):
    """Ignore lambda=1 and return remaining eigenvalues
    """
    w, _, _ = eig_transition_matrix(tmatrix)
    return w[1:]


def timescales(eigval, tau):
    return -tau / np.log(eigval)


def rayleigh_quotient(u, lagframes=1):
    """Parameters:
    ----------
    u : ndarray with shape (num_frames,)
        Some generic observable, u(x_t)
    lagframes : int
        How many frames to skip when computing correlation function

    Return:
    ------
    C(k tau) = E[u(x_t) u(x_{t + k tau})] / E[u(x_t) u(x_t)]
    """
    u0 = u[:-lagframes]
    uk = u[lagframes:]
    return (u0 * uk).mean() / (u0 * u0).mean()


def mfpt_matrix(counts, tau):
    """
    Return matrix of mean first passage times, M, where:
        M_ij = MFPT(i->j)

    Snell's formula
        M_ij = (Z_jj - Z_ij) / pi_j

    where
        Z = [1 - T + W]^{-1} and W_ij = pi_j
    """
    T = transition_matrix_mle(counts)
    pi = pi_mle(counts)

    zero = np.zeros(counts.shape)
    eye = np.eye(counts.shape[0])

    W = zero + pi
    Z = np.linalg.inv(eye - T + W)
    Z_w = zero + np.diag(Z)
    return tau * (Z_w - Z) / W


def sgrids(cvs, num_points=100, pad=0.07):
    """Parameters:
    ----------
    cvs : ndarray, shape (num_frames, num_cvs)
        The full dataset CV's
    num_points : int
        Number of points with which to represent grids
    pad : float
        The fraction of the range to extend grid on either end

    Return:
    s1, s1 : ndarrays, shape (num_points,)
    """
    def srange(s, pad):
        ds = pad * s.ptp()
        return s.min() - ds, s.max() + ds

    s1 = np.linspace(*srange(cvs[:, 0], pad), num_points)
    s2 = np.linspace(*srange(cvs[:, 1], pad), num_points)
    return s1, s2


def bias_dihedral_grid(cvs_grid, metad, num_gaussians=None):
    """Parameters:
    ----------
    cvs_grid : ndarray, shape (num_points, num_points)
        The value of the CV's at each dihedral value of dihedral grid
    metad : metad.Metadynamics
    num_gaussians : int
        How many Gaussians to use when computing w_t(s)?  If None, use them all

    Return:
    ------
    bias w(s), a ndarray with shape (num_points, num_points)
        The bias potential over the grid of dihedrals
    """
    num_rows, num_cols = cvs_grid.shape[:2]
    bias = np.full((num_rows, num_cols), np.nan)
    for i in range(num_rows):
        for j in range(num_cols):
            if not np.isnan(cvs_grid[i, j, 0]):
                s = cvs_grid[i, j, :]
                bias[i, j] = metad.bias_potential(s, num_gaussians=num_gaussians)
    return bias


def bias_cvs_grid(s1_grid, s2_grid, metad, num_gaussians=None):
    """Parameters:
    ----------
    s1_grid, s2_grid : ndarray, shape (num_points,)
    metad : metad.Metadynamics
    num_gaussians : int
        How many Gaussians to use when computing w_t(s)?  If None, use them all

    Return:
    ------
    bias w(s), a ndarray with shape (num_points, num_points)
        The bias potential over the grid of CV values
    """
    bias = np.zeros((len(s1_grid), len(s2_grid)))
    for i, s1 in enumerate(s1_grid):
        for j, s2 in enumerate(s2_grid):
            s = np.array([s1, s2])
            bias[i, j] = metad.bias_potential(s, num_gaussians=num_gaussians)
    return bias


def gaussian_centers_from_metad(fn, num_simulations):
    """Parameters:
    ----------
    fn : FileName
        File names for loading metad objects
    num_simulations : int
        How many of these files to use?

    Return:
    ------
    list of ndarrays, length num_simulations
        List of arrays of newly added Gaussian centers for each simulation, each has shape (num_new_centers, num_cvs)
    """
    metads = [load_npz(fn.metad_kde(frame)) for frame in range(num_simulations)]
    kde_kernel_counts = np.array([len(metad.centers) for metad in metads])

    metads = [load_npz(fn.metad(frame)) for frame in range(num_simulations)]
    centers = []
    for i in range(len(metads)):
        if i == 0:
            centers.append(metads[i].centers)
        else:
            # append only newly added Gaussians from this simulation
            centers.append(metads[i].centers[kde_kernel_counts[i-1]:])
    return centers


def macrostates_core(cvs):
    """Biased simulations spend too much time wandering around the transition states.  Need to use soft core state definitions.

    Parameters:
    ----------
    cvs : array with shape (num_frames, num_cvs)

    Return:
    ------
    macrostates : array (int), shape (num_frames,)
        Elements:
            0 = any 'eq' state that hasn't visited the 'ax' core
            1 = last found in 'ax' core
    """
    def eq_core(s1):
        return s1 > 0
    def ax_core(s1):
        return s1 < -5

    states = []
    for s1 in cvs[:, 0]:
        if len(states) == 0:
            if eq_core(s1):
                states.append(0)
            else:
                states.append(1)
        else:
            prev_state = states[-1]
            if prev_state == 1 and eq_core(s1):
                states.append(0)
            elif prev_state == 0 and ax_core(s1):
                states.append(1)
            else:
                states.append(prev_state)
    return np.array(states)


def transition_times(macrostates, lagtime, absolute=True):
    """
    Parameters:
    ----------
    macrostates : array (int), shape (num_frames,)
        Must be in {0, 1}
    lagtime : float
        Times separating frames (ns)
    absolute : bool
        If True, return times at which transitions occur
        Else, the times separating transitions
    """
    # transitions = -1 : 1 -> 0
    #                0 : 0 -> 0, 1 -> 1
    #                1 : 0 -> 1
    transitions = macrostates[:-1] - macrostates[1:]

    # count only 0 -> 1 transitions
    times = lagtime * (np.argwhere(transitions == 1) + 1).squeeze()
    if absolute:
        return times
    else:
        times = np.concatenate((np.array([0.0]), times))
        return (times[1:] - times[:-1])
