# WTMetaD simulation of alanine dipeptide with deep-TICA collective variables.

### WTMetaD:
A. Barducci, G. Bussi, and M. Parrinello, ``Well-tempered metadynamics: A smoothly converging and tunable free-energy method,'' *Phys. Rev. Lett.*, **100**, 020603 (2008)


### Deep TICA:
W. Chen, H. Sidky, and A. L. Ferguson, ``Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets,'' *J. Chem. Phys.*, **150**, 214114 (2019)


### OpenMM:

Networks are created and trained with PyTorch:
https://pytorch.org/

Simulations are done using OpenMM
https://openmm.org/

with bias forces implemented by the PyTorch plugin:
https://github.com/openmm/openmm-torch

and converted to a TorchScript module.  The conda environment is described in `openmm-torch.yml`.


