import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import Dataset
from .data import TimeLaggedDataset, TrajectoryDataset
from .util import to_torch


def sym_eig(a: torch.Tensor):
    """Solve (regularized) eigenvalue problem for symmetric torch Tensor
    ** TODO: clearer to put transpose in sym_inverse rather than here?
    ** TODO: surely pytorch can infer dtype, device?
    """
    ar = a + 1e-6 * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    f = torch.linalg.eigh(ar)
    return torch.abs(f.eigenvalues), f.eigenvectors.t()


def sym_inverse(a: torch.Tensor, return_sqrt=False):
    eigvals, eigvecs = sym_eig(a)
    if return_sqrt:
        eigvalm = torch.diag(torch.sqrt(1 / eigvals))
    else:
        eigvalm = torch.diag(1 / eigvals)
    return eigvecs.t() @ eigvalm @ eigvecs


def cov_matrices(x: torch.Tensor, y: torch.Tensor):
    c00 = 1 / (x.shape[0] - 1) * (x.t() @ x)
    c11 = 1 / (y.shape[0] - 1) * (y.t() @ y)
    c01 = 1 / (x.shape[0] - 1) * (x.t() @ y)
    c0 = 0.5 * (c00 + c11)      # average x, y variance
    c1 = 0.5 * (c01 + c01.t())  # add reverse transitions
    return c0, c1


def koopman_matrix(x: torch.Tensor, y: torch.Tensor):
    """Minibatch estimate of Koopman matrix during training.  The subtraction of the means of x, y vectors means that the eigenfunction corresponding to the equilibrium process has been projected out.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    c0, c1 = cov_matrices(x, y)
    inv_sqrt_c0 = sym_inverse(c0, return_sqrt=True)
    return inv_sqrt_c0 @ c1 @ inv_sqrt_c0


class VAMPNet:
    """Optimize objective function of Koopman matrix, K
    vamp1: loss = -(1 + tr K), where the 1 is the Perron eigenvalue
    vamp2: loss = -(1 + tr KK')
    """
    def __init__(self, net, device=None, learning_rate=1e-3, loss_method='vamp2'):
        assert loss_method in ('vamp1', 'vamp2')
        self.net = net.to(device=device).float()
        self.device = device
        self.optim = torch.optim.Adam(params=self.net.parameters(),
                                      lr=learning_rate)
        self.loss_method = loss_method

        self._train_scores = []
        self._test_scores = []

    @property
    def train_scores(self):
        return np.array(self._train_scores)

    @property
    def test_scores(self):
        return np.array(self._test_scores)

    def fit(self, data_loader_train, data_loader_test, num_epochs=1, progress=None):
        for epoch in progress(range(num_epochs), desc="VAMPnet epoch",
                              total=num_epochs, leave=False):
            # training
            self.net.train()
            for x, y in data_loader_train:
                self.optim.zero_grad()
                loss = self.loss(self.net(x.to(device=self.device)),
                                 self.net(y.to(device=self.device)))
                loss.backward()
                self.optim.step()
                self._train_scores.append([epoch, (-loss).item()])

            # validation
            self.net.eval()
            for x, y in data_loader_test:
                loss = self.loss(self.net(x.to(device=self.device)),
                                 self.net(y.to(device=self.device)))
                self._test_scores.append([epoch, (-loss).item()])

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        koopman = koopman_matrix(x, y)
        if self.loss_method == 'vamp1':
            vamp_score = torch.norm(koopman, p='nuc')
        else:
            vamp_score = torch.square(torch.norm(koopman, p='fro'))
        return -(1 + vamp_score)


class SRV:
    """Network:
    - references are vampnet.net, self.net, returned reference in srv_net()
    - changed to inference mode in __init__()
    - moved to CPU in fit()

    Eigenfunctions:
    SRV.__call__(x) returns ndarray, shape (num_eigvecs,)
    srv_net(x) returns ndarray, shape (num_cvs,)

    Pytorch module notes:
    ---------------------
    1. Check device network is on:
        next(net.parameters()).device

    1. Train, evaluation modes
        net.train() -> net.training = True
        net.eval() -> net.training = False
    Evaluation mode: ignores dropouts, batchnorm taken from saved statistics and not computed on the fly.

    2. Disabling autograd: when you don't need/want to track gradients on parameters
        context manager:    with torch.no_grad():
        set to inference:   for p in net.parameters():
                                p.requires_grad = False

    TODO:  some of these methods should be @classmethod
    """
    def __init__(self, net, tau):
        """
        Paramters:
        ----------
        net : torch.nn.modules.container.Sequential
            From vampnet.net
        tau : float
            Lagtime in ps
        """
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.tau = tau
        self.num_eigvecs = self.net[-2].out_features
        self.device = self.net[-2].bias.device
        self.mean = None
        self.transform_matrix = None
        self.eigvals = None

    def timescales(self, tau):
        return -self.tau / np.log(self.eigvals)

    def fit(self, dataset):
        from scipy.linalg import inv, sqrtm

        assert type(dataset) in (TimeLaggedDataset, TrajectoryDataset)
        if type(dataset) is TrajectoryDataset:
            transform = self._transform_trajectory_dataset
        elif type(dataset) is TimeLaggedDataset:
            transform = self._transform_timelagged_dataset

        z0, z1, mean = transform(dataset)
        c0, c1 = cov_matrices(z0, z1)

        # move to numpy float64's for algebra
        self.mean = self._to_numpy_double(mean)
        c0 = self._to_numpy_double(c0)
        c1 = self._to_numpy_double(c1)
        inv_sqrt_c0 = inv(sqrtm(c0))
        koopman = (inv_sqrt_c0 @ c1 @ inv_sqrt_c0)
        self.eigvals, eigvecs = self._eigh_sorted(koopman)
        self.transform_matrix = inv_sqrt_c0 @ eigvecs

    def srv_net(self, num_cvs=2):
        """Return network with transformation added as a linear layer.
        The layers are set to inference mode, and moved to CPU.
        """
        W = to_torch(self.transform_matrix[:, :num_cvs])
        b = -to_torch((self.mean @ self.transform_matrix)[:, :num_cvs])

        # torch convention for layer: y = xW' + b
        eig_layer = nn.Linear(self.num_eigvecs, num_cvs)
        eig_layer.weight = nn.Parameter(W.t())
        eig_layer.bias = nn.Parameter(b)

        net = nn.Sequential(*self.net, eig_layer)
        for p in net.parameters():
            p.requires_grad = False
        return net.to(device=torch.device('cpu')).float()

    def _transform_torch(self, features):
        return self.net(to_torch(features, device=self.device))

    def _transform_trajectory_dataset(self, dataset):
        """Parameters:
        ----------
        dataset : src.data.TrajectoryDataset

        Return:
        ------
        z0 : torch.Tensor, shape (num_frames, num_eigvecs)
        z1 : torch.Tensor, shape (num_frames, num_eigvecs)
            This is the transformed data and data_lagged.  It is zero-mean
        mean : torch.Tensor, shape (1, num_eigvecs)
            The mean used to shift data.  Required for the TICA transform.
        """
        zt = self._transform_torch(dataset.trajectory)
        mean = zt.mean(dim=0, keepdim=True)
        zt -= mean
        return zt[:-dataset.lagframes], zt[dataset.lagframes:], mean

    def _transform_timelagged_dataset(self, dataset):
        """Parameters:
        ----------
        dataset : src.data.TimeLaggedDataset

        See above.
        """
        z0 = self._transform_torch(dataset.data)
        z1 = self._transform_torch(dataset.data_lagged)
        mean = 0.5 * z0.mean(dim=0, keepdim=True) \
                + 0.5 * z1.mean(dim=0, keepdim=True)
        return z0 - mean, z1 - mean, mean

    def _to_numpy_double(self, a):
        return a.cpu().numpy().astype('float64')

    def _eigh_sorted(self, a):
        w, v = np.linalg.eigh(a)
        idx = w.argsort()[::-1]
        return np.real(w[idx]), np.real(v[:, idx])

    def __call__(self, features):
        z = self.net(to_torch(features, device=self.device)).cpu().numpy()
        return (z - self.mean) @ self.transform_matrix
