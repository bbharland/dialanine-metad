import numpy as np
import mdtraj as md
import mdshare
import os

from .util import to_torch


class TimeLaggedDataset:
    r"""deeptime/util/data.py """

    def __init__(self, data, data_lagged):
        """Parameters:
        ----------
        data : ndarray with shape (num_frames - lagframes, num_features)
            Molecular configurations of {x_t}
        data_lagged : ndarray with shape (num_frames - lagframes, num_features)
            Molecular configurations of {x_{t+tau}}
        """
        assert len(data) == len(data_lagged), \
            f"Length mistmatch: {len(data)=} != {len(data_lagged)=}"
        self.data = data
        self.data_lagged = data_lagged

    def astype(self, dtype):
        return TimeLaggedDataset(self.data.astype(dtype),
                                 self.data_lagged.astype(dtype))

    def __getitem__(self, item):
        return self.data[item], self.data_lagged[item]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        data = np.vstack((self.data, other.data))
        data_lagged = np.vstack((self.data_lagged, other.data_lagged))
        return TimeLaggedDataset(data, data_lagged)


class TrajectoryDataset(TimeLaggedDataset):
    r"""deeptime/util/data.py """

    def __init__(self, lagframes, trajectory):
        """Parameters:
        ----------
        lagframes : int
            Number of simulation (.h5) frames separating transition
        trajectory : ndarray with shape (num_frames, num_features)
            Feature data from simulation
        """
        assert lagframes > 0, "lagframes must be positive"
        assert len(trajectory) > lagframes, "Not enough data to for lagtime"
        self.lagframes = lagframes
        self.trajectory = trajectory
        super().__init__(trajectory[:-lagframes], trajectory[lagframes:])


class SimulationData:
    """Provide handles for things computed from trajectories.

    The sequence is:
        1. Get dihedrals, features out of a trajectory
        2. Train network with features
        3. Save eigenfunctions and CVs for trajectory
        4. Project trajectory eigenfunctions/CVs onto dihedral grid
    """
    def __init__(self, working_dir, lagtime, lagframes):
        """
        Parameters
        ----------
        working_dir: str
            Location for data files to be written
        lagtime: float
            Time separating transitions (tau, in ps)
        lagframes: int
            Number of simulation frames that make up this lagtime
        """
        self.dir = working_dir
        self.lagtime = lagtime
        self.lagframes = lagframes
        self.data_dict = dict()

    @property
    def dataset(self):
        return TrajectoryDataset(lagframes=self.lagframes,
                                 trajectory=self.features)

    def fetch(self, name):
        if name not in self.data_dict:
            filename = f'{self.dir}/{name}.npy'
            if os.path.isfile(filename):
                self.data_dict[name] = np.load(filename)
            else:
                raise FileNotFoundError(f'{filename} not found')
        return self.data_dict[name]

    @property
    def dihedrals(self):
        return self.fetch('dihedrals')

    @property
    def features(self):
        return self.fetch('features')

    @property
    def cvs(self):
        return self.fetch('cvs')

    @property
    def eigvals(self):
        return self.fetch('eigvals')

    @property
    def timescales(self):
        return self.fetch('timescales')

    @property
    def psi(self):
        return self.fetch('psi')

    @property
    def theta_grid(self):
        return self.fetch('theta_grid')

    @property
    def psi_grid(self):
        return self.fetch('psi_grid')

    @property
    def cvs_grid(self):
        return self.fetch('cvs_grid')

    def save_feature_data(self):
        """Save files if they don't exist:

        dihedrals.npy : ndarray with shape (num_frames, 2)
        features.npy : ndarray with shape (num_frames, num_features)
        """
        raise NotImplementedError("Don't use base class")

    def save_eigen_data(self, srv, num_cvs=2):
        """Call once the SRV object has been fitted.  Save files:

        eigvals.npy : ndarray with shape (num_eigvals,)

        timescales.npy : ndarray with shape (num_eigvals,)
            Timescales corresponding to eigenvalues (sorted)

        psi.npy : ndarray with shape (num_frames, num_eigvals)
            Transformed dataset

        cvs.npy : ndarray with shape (num_frames, num_cvs)

        Parameters
        ----------
        srv : vampnet.SRV
        """
        psi = srv(self.features)
        np.save(f'{self.dir}/eigvals.npy', srv.eigvals)
        np.save(f'{self.dir}/timescales.npy', srv.timescales(self.lagtime))
        np.save(f'{self.dir}/psi.npy', psi)
        np.save(f'{self.dir}/cvs.npy', psi[:, :num_cvs])

    def save_dihedral_grid_data(self, num_cvs=2, num_points=100):
        """Requires psi.  Save files:

        theta_grid.npy : ndarray with shape (num_points,)
            Grid values of phi, psi on [-np.pi, +np.pi]

        psi_grid.npy : ndarray with shape (num_points, num_points, num_eigvecs)
            The values of the eigenfunctions for each grid point

        cvs_grid.npy : ndarray with shape (num_points, num_points, num_cvs)
            The values of the CVs for each grid point

        Parameters
        ----------
        num_points : number of grid points for each diheral angle
        """
        def grid_ndx(theta, dtheta):
            return round((theta + np.pi) / dtheta)

        num_eigvecs = self.psi.shape[1]
        theta_grid = np.linspace(-np.pi, np.pi, num_points)
        dtheta = theta_grid[1] - theta_grid[0]
        psi_grid = np.full((num_points, num_points, num_eigvecs), np.nan)

        for dihedral, psi in zip(self.dihedrals, self.psi):
            i = grid_ndx(dihedral[0], dtheta)
            j = grid_ndx(dihedral[1], dtheta)
            if np.isnan(psi_grid[i, j, 0]):
                psi_grid[i, j, :] = psi

        np.save(f'{self.dir}/theta_grid.npy', theta_grid)
        np.save(f'{self.dir}/psi_grid.npy', psi_grid)
        np.save(f'{self.dir}/cvs_grid.npy', psi_grid[:, :, :num_cvs])

    def heavy_atom_distance_features(self, traj, periodic=True):
        """Return : ndarray with shape (num_frames, num_pairs)
            The 45 atom-pair distances for an mdtraj Trajectory
        """
        heavy_atoms = traj.topology.select('resname != HOH && type != H')
        atom_pairs = np.array([(i, j)
                               for i in heavy_atoms
                               for j in heavy_atoms if i < j])
        return md.compute_distances(traj, atom_pairs, periodic=periodic)


class MDShareData(SimulationData):
    """ https://markovmodel.github.io/mdshare/ALA2/#alanine-dipeptide
    """
    def __init__(self, working_dir):
        super().__init__(working_dir, lagtime=1.0, lagframes=1)

    def save_feature_data(self):
        """Save files if they don't exist:

        dihedrals.npy : ndarray with shape (num_frames, 2)
        features.npy : ndarray with shape (num_frames, num_features)
        """
        fn = 'alanine-dipeptide-nowater.pdb'
        top = mdshare.fetch(fn, working_directory=self.dir)

        fn = 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz'
        with np.load(mdshare.fetch(fn, working_directory=self.dir)) as fh:
            dihedrals = np.vstack([fh[f"arr_{i}"] for i in range(3)])
        np.save(f'{self.dir}/dihedrals.npy', dihedrals)

        fns = [f'alanine-dipeptide-{i}-250ns-nowater.xtc' for i in range(3)]
        xtc_files = [mdshare.fetch(fn, working_directory=self.dir) for fn in fns]

        fn = f'{self.dir}/features.npy'
        if not os.path.isfile(fn):
            features = np.vstack([self.features_from_xtc(f, top) for f in xtc_files])
            np.save(fn, features)

    def features_from_xtc(self, xtc_file, top):
        traj = md.load_xtc(xtc_file, top=top)
        return self.heavy_atom_distance_features(traj, periodic=False)


class OpenMMData(SimulationData):
    def __init__(self, working_dir, lagtime=1.0, lagframes=1):
        super().__init__(working_dir, lagtime, lagframes)

    def save_feature_data(self, fn_h5, num_simulations, periodic=True, recalculate=False):
        """Save files if they don't exist (or recalculate is set to True):

        dihedrals.npy : ndarray with shape (num_frames, 2)
        features.npy : ndarray with shape (num_frames, num_features)

        Parameters:
        ----------
        fn_h5 : function : int -> str
            Give the path to the .h5 file containing trajectory segment
        """
        fn = f'{self.dir}/dihedrals.npy'
        if not os.path.isfile(fn) or recalculate:
            print('writing', fn)
            dihedrals = []
            for sn in range(num_simulations):
                traj = md.load(fn_h5(sn))
                dihedral = np.hstack([md.compute_phi(traj)[1],
                                      md.compute_psi(traj)[1]])
                dihedrals.append(dihedral)
            dihedrals = np.vstack(dihedrals)
            np.save(fn, dihedrals)
        else:
            print('not recalculating', fn)

        fn = f'{self.dir}/features.npy'
        if not os.path.isfile(fn) or recalculate:
            print('writing', fn)
            features = np.vstack([
                self.features_from_traj(fn_h5(sn), periodic=periodic)
                for sn in range(num_simulations)
            ])
            np.save(fn, features)
        else:
            print('not recalculating', fn)

    def features_from_traj(self, traj_file, periodic=True):
        traj = md.load(traj_file)
        return self.heavy_atom_distance_features(traj, periodic=periodic)

