import numpy as np
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix
from taps.ml.torch import TensorProjector
from ase import Atoms


class DScribe(TensorProjector):
    """

    # Define atomic structures
    samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]

    # Setup descriptors
    cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
    soap_desc = SOAP(species=["C", "H", "O", "N"], rcut=5, nmax=8, lmax=6,
    crossover=True)
    # Create descriptors as numpy arrays or sparse arrays
    water = samples[0]
    coulomb_matrix = cm_desc.create(water)
    soap = soap_desc.create(water, positions=[0])

    # Easy to use also on multiple systems, can be parallelized across processes
    coulomb_matrices = cm_desc.create(samples)
    coulomb_matrices = cm_desc.create(samples, n_jobs=3)
    oxygen_indices = [np.where(x.get_atomic_numbers() == 8)[0] for x in samples]
    oxygen_soap = soap_desc.create(samples, oxygen_indices, n_jobs=3)

    # Some descriptors also allow calculating derivatives with respect to atomic
    # positions
    der, des = soap_desc.derivatives(samples, method="auto", return_descriptor=True)

    Nd = 100
symbols = ['H', 'O', 'H']
species = list(set(symbols))
prj = DScribeProjector(symbols=symbols,
                    generator=SOAP(species=species, rcut=5, nmax=8, lmax=6, crossover=True))
aa =  AtomCenteredPyTorchKernel(symbols=symbols, Nd=Nd, prj=prj)



    """
    def __init__(self, symbols, generator=None):
        # Setup descriptors
        self.symbols = symbols
        self.generator = generator
        self.sort = np.argsort(symbols)

    def x(self, coords):
        D, N = coords.D, coords.N
        images = []
        for positions in coords.coords.T:
            images.append(Atoms(self.symbols,
                                positions=positions))
        # NxAxdesc
        desc = self.generator.create(images)
        return tc.from_numpy(desc[:, self.sort, :].reshape(N, -1)).requires_grad_(True).double()
