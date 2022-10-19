import numpy as np
import torch
import openmm.unit as unit


def metadynamics(temperature, bias_factor, height, width):
    """Deal with units and return a Metadynamics object.

    Parameters
    ----------
    temperature : openmm.unit.quantity.Quantity
        Temperature the simulation is run at, in K.

    bias_factor : float
        For WTMetaD, deltaT = T * (bias_factor - 1)

    height : openmm.unit.quantity.Quantity
        Standard height of biasing Gaussians (will convert to kJ/mol).

    width : np.ndarray with shape (num_cvs,)
        The Gaussian widths along the direction of each CV.
    """
    assert unit.is_quantity(temperature), "temperature must be a unit in K"
    assert bias_factor > 1, f"{bias_factor = } must be greater than 1"
    assert unit.is_quantity(height), "height must be a unit"
    assert type(width) == np.ndarray and len(width.shape) == 1, \
    "width must be 1d np.ndarray"

    deltaT = temperature * (bias_factor - 1)
    betap = 1 / (unit.MOLAR_GAS_CONSTANT_R * deltaT)
    betap = betap.in_units_of(unit.mole / unit.kilojoule)._value
    height = height.in_units_of(unit.kilojoule / unit.mole)._value
    return Metadynamics(betap, height, width)


class Metadynamics:
    """Reference:  Valsson, Tiwary, and Parrinello, "Enhancing Important Fluctuations: Rare Events and Metadynamics from a Conceptual Viewpoint", Annu. Rev. Phys. Chem. 67, 159 (2016) https://www.annualreviews.org/doi/abs/10.1146/annurev-physchem-040215-112229
    """
    def __init__(self, betap, height, width):
        """
        betap : float
            Bias potential tempering factor

        height : float
            Base height for Gaussians (in kJ/mol)

        width : np.ndarray with shape (num_cvs,)
            The Gaussian widths along the direction of each CV.
        """
        self._betap = betap
        self._height = height
        self._width = width
        self.heights = np.empty(0, dtype=np.float32)
        self.centers = np.empty((0, len(width)), dtype=np.float32)
        self.widths = np.empty((0, len(width)), dtype=np.float32)

    def __len__(self):
        return len(self.heights)

    def add_gaussian(self, sn):
        """Add a new Gaussian centered at sn : ndarray with shape (num_cvs,)
        """
        height = np.array(
            self._height * np.exp(-self._betap * self.bias_potential(sn))
        )
        self.heights = np.hstack([self.heights, height])
        self.centers = np.vstack([self.centers, sn])
        self.widths = np.vstack([self.widths, self._width])

    def bias_potential(self, s, num_gaussians=None):
        """Return w(s).  If num_gaussians is not None, return bias potential using only this many terms.
        """
        if len(self) == 0:
            return 0.0
        if num_gaussians is None:
            num_gaussians = len(self)
        h = self.heights[:num_gaussians]
        c = self.centers[:num_gaussians]
        w = self.widths[:num_gaussians]
        return np.sum(h * np.exp(-0.5 * np.sum(((s - c) / w)**2, axis=1)))

    def force_module(self, net):
        """Factory function for openmm-torch ForceModule

        Parameters
        ----------
        net : torch.nn.modules.container.Sequential
            The network that transforms features -> collective variables.  This needs to be in inference mode (i.e. all parameters are set to requires_grad = False)

        Returns
        -------
        ForceModule(torch.nn.Module)
            The module to be compiled and added to the simulation.
        """
        assert len(self) > 0, "Can't deal with empty metad object"
        dtype = torch.float32
        width = torch.tensor(self._width, dtype=dtype).unsqueeze(0)

        heights = torch.tensor(self.heights, dtype=dtype)
        centers = torch.tensor(self.centers, dtype=dtype)
        widths = torch.tensor(self.widths, dtype=dtype)
        return ForceModule(net, heights, centers, width, widths)


class ForceModule(torch.nn.Module):
    """Dialanine module for metadynamics in openmm with latent variables.
        * heavy-atom distance featurization (45 features)
        * two latent variables (num_cvs = 2)

        Autodiff only cares about input (positions) and output (bias_potential).  Everything else must be a buffer.

        TODO: Figure out a way to turn off autodiff during 'add_gaussian()'?  Until then, Metadynamics object has to handle this.

        TODO: Figure out a way to initialize with no Gaussians?
    """
    def __init__(self, net, heights, centers, width, widths):
        """
        Parameters
        ----------
        net : torch.nn.modules.container.Sequential
            The network that transforms features -> collective variables.  This must be in inference mode for autodiff to work properly.

        heights : torch.Tensor with shape (num_gaussians,)
            Heights of Gaussians in kJ/mol

        centers : torch.Tensor with shape (num_gaussians, num_cvs)
            Locations of Gaussians

        width : torch.Tensor with shape (1, num_cvs)
            Width along each CV for newly deposited Gaussians.

        widths : torch.Tensor with shape (num_gaussians, num_cvs)
            These are identical for all Gaussians until a KDE compression is done.
        """
        super().__init__()
        self.net = net
        self.register_buffer('heights', heights)
        self.register_buffer('centers', centers)
        self.register_buffer('width', width)
        self.register_buffer('widths', widths)

    def forward(self, positions, add_gaussian, height, center1, center2):
        """Calculate the bias potential energy from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles, 3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        add_gaussian : torch.Scalar, cast to bool by torch.jit
            Global parameters are handled by the TorchForce swig object:
                torch_force.addGlobalParameter('add_gaussian', False)
            and modified from the context:
                simulation.context.setParameter('add_gaussian', True)

        If add_gaussian = True, use values below for new Gaussian
        height : torch.Scalar = torch.Tensor with shape ()
        center1 : torch.Scalar = first component of center, s1
        center2 : torch.Scalar = second component, s2

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        if add_gaussian:
            device = self.heights.device
            h = height.to(device=device)
            c = torch.stack([center1, center2]).to(device=device)
            self.heights = torch.cat([self.heights, h.unsqueeze(0)], dim=0)
            self.centers = torch.cat([self.centers, c.unsqueeze(0)], dim=0)
            self.widths = torch.cat([self.widths, self.width], dim=0)

        x = self.featurize(positions)
        s = self.net(x)
        return self.bias_potential(s)

    def featurize(self, positions):
        """Return x : torch.Tensor with shape (1, num_features)
            This is the input for the network
        """
        return torch.stack([
            torch.sqrt(torch.sum(torch.square(positions[j] - positions[i])))
            for (i, j) in ((1, 4), (1, 5), (1, 6), (1, 8), (1, 10),
                           (1, 14), (1, 15), (1, 16), (1, 18), (4, 5),
                           (4, 6), (4, 8), (4, 10), (4, 14), (4, 15),
                           (4, 16), (4, 18), (5, 6), (5, 8), (5, 10),
                           (5, 14), (5, 15), (5, 16), (5, 18), (6, 8),
                           (6, 10), (6, 14), (6, 15), (6, 16), (6, 18),
                           (8, 10), (8, 14), (8, 15), (8, 16), (8, 18),
                           (10, 14), (10, 15), (10, 16), (10, 18), (14, 15),
                           (14, 16), (14, 18), (15, 16), (15, 18), (16, 18))
            ]).unsqueeze(0)

    def bias_potential(self, s):
        """
        Parameters
        ----------
        s : torch.Tensor with shape (1, num_cvs)
           The values of the collective variables

        Returns
        -------
        potential : torch.Scalar
           The bias potential energy (in kJ/mol)
        """
        norm_sqs = torch.sum(torch.square((s - self.centers) / self.widths), 1)
        return torch.sum(self.heights * torch.exp(-0.5 * norm_sqs))

    # def add_gaussian(self, sn):
    #     """TODO:  This is where turning off the trace would allow this code to do it's own metadynamics.  Something like:
    #     """
    #     with torch.no_grad():
    #         h = self.heights[0] * torch.exp(-self.betap * self.bias_potential(sn))
    #     self.centers = torch.cat([self.centers, sn], dim=0)
    #     self.heights = torch.cat([self.heights, h.unsqueeze(0)], dim=0)
    #     self.widths = torch.cat([self.widths, self.width], dim=0)


class Gaussian:
    """Class for adding Gaussians and computing Mahalanobis distances between them.
    """
    def __init__(self, height, center, width):
        self.h = height # float
        self.c = center # ndarray with shape (num_cvs,)
        self.w = width  # ndarray with shape (num_cvs,)

    # def __call__(self, s):
    #     return self.h * np.exp(-0.5 * self.distance(s)**2)

    def __add__(self, other):
        height = self.h + other.h
        center = (self.h * self.c + other.h * other.c) / height
        ws =  self.h * (self.w**2 + self.c**2)
        wo =  other.h * (other.w**2 + other.c**2)
        width = np.sqrt((ws + wo) / height - center**2)
        return Gaussian(height, center, width)

    def distance(self, s):
        return np.sqrt(np.sum(((s - self.c) / self.w)**2))


def kde_compression(metad, dist_threshold=1):
    """Reference: Supplemantary Information for Michele Invernizzi, Pablo M. Piaggi, and Michele Parrinello, "Unified Approach to Enhanced Sampling", Phys. Rev. X 10, 041034 (2020), https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041034
    """
    gaussians = []

    for h, c, w in zip(metad.heights, metad.centers, metad.widths):
        gn = Gaussian(h, c, w)
        keep_merging = True

        while keep_merging:
            if len(gaussians) == 0:
                gaussians.append(gn)
                keep_merging = False
            else:
                dists = [g.distance(gn.c) for g in gaussians]
                idx = np.argmin(dists)
                if dists[idx] > dist_threshold:
                    gaussians.append(gn)
                    keep_merging = False
                else:
                    gn += gaussians[idx]
                    del gaussians[idx]

    new_metad = Metadynamics(metad._betap, metad._height, metad._width)
    new_metad.heights = np.array([g.h for g in gaussians])
    new_metad.centers = np.vstack([g.c for g in gaussians])
    new_metad.widths = np.vstack([g.w for g in gaussians])
    return new_metad
