import os


class FileNames:
    def __init__(self, working_dir, ns):
        self.working_dir = working_dir
        self.ns = ns
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

    def filename(self, prefix, file_ext, suffix=None, directory=None):
        """Return file names with format:
            <directory>/<prefix>_<ns>ns_<suffix><.ext>
        """
        if directory is None:
            filename = f'{self.working_dir}/{prefix}_{self.ns}ns'
        else:
            filename = f'{directory}/{prefix}_{self.ns}ns'
        if suffix is not None:
            filename += f'_{suffix}'
        return filename + file_ext

    def h5(self, sim_num):
        return self.filename('ala2', '.h5', suffix=sim_num)

    def out(self, sim_num):
        return self.filename('ala2', '.out', suffix=sim_num)

    def metad(self, sim_num):
        return self.filename('metad', '.npz', suffix=sim_num)

    def metad_kde(self, sim_num):
        return self.filename('metad', '.npz', suffix=f'kde_{sim_num}')

    def final_positions(self, sim_num):
        return self.filename('final_positions', '.npz', suffix=sim_num)

    def bias_grid(self, sim_num):
        return self.filename('bias_grid', '.npy', suffix=sim_num)

    def bias_grid_kde(self, sim_num):
        return self.filename('bias_grid', '.npy', suffix=f'kde_{sim_num}')

    def dihedral(self, sim_num):
        return self.filename('dihedral', '.npy', suffix=sim_num)

    def cvs_traj(self, sim_num):
        return self.filename('cvs_traj', '.npy', suffix=sim_num)

    def bias_traj(self, sim_num):
        return self.filename('bias_traj', '.npy', suffix=sim_num)
