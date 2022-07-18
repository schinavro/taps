
import torch as tc
from torch import nn

import numpy as np
from ase import Atoms


class Descriptor(nn.Module):
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

    def __init__(self, species=None, **kwargs):
        super(Descriptor, self).__init__()
        self.species = species

    def forward(self, tensor):
        N, A, _ = tensor.shape
        temp = []
        for a in range(A):
            temp.append(tensor - tensor[:, a, None])
        return tc.cat(temp, axis=2)

    def x(self, tensor):
        """
        tensor: N x A x 3
        return: N x A x 3A
        """
        return self.forward(tensor)


class Naive(nn.Module):
    def __init__(self, **kwargs):
        super(Naive, self).__init__()

    def forward(self, tensor):
        N, A, _ = tensor.shape
        temp = []
        for a in range(A):
            temp.append(tensor - tensor[:, a, None])
        return tc.cat(temp, axis=2)

    def x(self, tensor):
        """
        tensor: N x A x 3
        return: N x A x 3A
        """
        return self.forward(tensor)


def get_neighbors_info(pnl=None, positions=None, cell=None, pbc=None,
                       cutoff=None):
    """
    Get neighbors info

    Parameters
    ----------

    positions: tensor array
      Atomic positions
    cell: tensor (3, 3)
      Periodic boundary
    pbc: Tuple of bool
      Periodic boundary condition

    Return
    ------

    (iidx, jidx, disp, dist)
    iidx: tensor of long int (NM, ) index of i
    jidx: tensor of long int (NM, ) index of j
    disp: tneosr of float (NM, 3) displacements rj - ri
    dist: norm of disp (NM, )
    """
    if pnl is None:
        # from ase.neighborlist import primitive_neighbor_list
        from taps.utils.neighborlist import primitive_neighbor_list
        nl = primitive_neighbor_list(pbc=pbc, cell=cell,
                                     positions=positions,
                                     cutoff=cutoff,
                                     self_interaction=False, quantities="ijDd")
        return nl
    pnl.update(pbc, cell, positions)

    iidx = []
    for i in range(len(positions)):
        iidx.extend([i] * len(pnl.neighbors[i]))

    jidx = tc.cat(pnl.neighbors)
    disp = (positions[jidx] + tc.vstack(pnl.displacements).double() @ cell) - positions[iidx]
    dist = tc.linalg.norm(disp, axis=1)

    if isinstance(positions, np.ndarray):
        return iidx, pnl.neighbors, np.array(disp), np.array(dist)
    else:
        return tc.Tensor(iidx).long().to(device=cell.device), jidx, disp, dist


def sort_atomic_numbers(numbers):
    device = numbers.device
    species = list(set(numbers.tolist()))
    sorter = dict([(spe, i) for i, spe in enumerate(species)])
    srtd_n = [sorter[n] for n in numbers.tolist()]
    srtd_s = [sorter[s] for s in species]
    return tc.tensor(species, device=device), tc.tensor(srtd_s, device=device), tc.tensor(srtd_n, device=device)


def cutoff(dist, rcut):
    """
    dist: NNTot arr
    rcut: float
    return NNTot arr
    """
    # return tc.ones(dist.shape, dtype=dist.dtype, device=dist.device)
    return 0.25 * (tc.cos(dist * np.pi/rcut)+1)**2


def gauss(dist, jatn, alpha, rs):
    """
    dist : NN
    sig : S x nmax
    rs : S x nmax
    return : NN x nmax
    """

    """
    NN = len(dist)
    dtype, device = dist.dtype, dist.device
    gauss = tc.empty((NN, nmax), dtype=dtype, device=device)
    for i, spe in enumerate(species):
        # nmax
        Sig, Rs = sigma[i], rs[i]
        mask = (numbers == spe)
        # NN x nmax
        tmp = np.exp(Sig * (dist[mask, None] - Rs)**2)
        # NN x 1
        gauss.masked_scatter_(mask[:, None], tmp)
    return gauss
    """
    Rs = rs[jatn]
    alp = alpha[jatn]
    return tc.exp(alp * (dist[:, None] - Rs)**2)


def angular(disp, lmax, NO, dtype, device):
    """
    unit : NN x 3
    disp : NN x 3
    fcut : NN
    lmax : int
    return : NN x O
       where NN is number of neighbor
              O is number of orbitals

    Only 1, 2 support.
    """
    NN = len(disp)                     # Number of Neighboribng images

    angular = tc.ones(NN, NO, dtype=dtype, device=device)
    if lmax > 1:
        # NNx1 * NNx3 -> NNx3
        angular[:, 1:4] = disp
    if lmax > 2:
        pass
        """
        for l in range(2, lmax+1):
            for m in range(l):
                Tlm = chebyshev(l, m)
                for pqr in mulinomial(m):
                    p, q, r = pqr
        """
    return angular


def get_density(Wln, Csn, Fxyz, iidx, jidx, device, dtype, NA, NO, nmax):
    """
    Wln: O x nmax x O
    Csn: NA x nmax
    Fxyz: NNxNOxnmax
    Return NA x O
    """
    # NAxnmax -> NNxnmax
    cj = Csn[jidx]
    # NNx1xnmax x NNxNOxnmax -> NNxNOxnmax
    cjFxyz = cj[:, None] * Fxyz

    # NN x NO x nmax -> NA x NO x nmax
    bnl = tc.zeros((NA, NO, nmax), device=device, dtype=dtype).index_add(
                                            0, iidx, cjFxyz)

    # NOx(nmax)xNO * NAxNOx(nmax)x1, -> NA x NO x nmax x NO -> NAxOxO -> NAxO
    return tc.sum(tc.sum(Wln * bnl[..., None], axis=2) ** 2, axis=1)


class Gj(nn.Module):
    def __init__(self, moduledict, numbers=None, nmax=None):
        super(Gj, self).__init__()
        self.moduledict = moduledict
        self.numbers = numbers
        self.NA = len(numbers)
        self.nmax = nmax

    def forward(self, ρ):
        """
        ρ : NA x O
        return : NA x nmax
        """
        coeff = tc.zeros((self.NA, self.nmax), dtype=ρ.dtype, device=ρ.device)
        for n, spe in enumerate(self.numbers):
            coeff[n] = self.moduledict[str(spe)](ρ[n])
        return coeff


class REANN(nn.Module):
    """
    Examples
    --------
    from torch import nn
    from taps.ml.descriptors.torch import REANN
    A = len(init)
    pbc = [True, True, True]
    cutoffs = [6.] * A

    desc = REANN(pbc=pbc, cell=cell, cutoffs=cutoffs, numbers=numbers)
    from ase.neighborlist import PrimitiveNeighborList
        pnl = PrimitiveNeighborList(cutoffs=[6]*len(init),
                                    self_interaction=False, bothways=True)
        pnl.build(pbc, init.cell, init.positions)
    """
    def __init__(self, pnl=None, pbc=None, cell=None, cutoffs=None,
                 numbers=None,
                 rcut=6., lmax=2, nmax=2,
                 loop=1, **kwargs):
        super(REANN, self).__init__()

        self.pnl = pnl or self.primitive_neighbor_list(cutoffs)
        self.pbc = tc.tensor(pbc).to(device=cell.device)
        self.cell = cell
        # self.cutoff = cutoff
        self.numbers = numbers
        _, __, ___ = sort_atomic_numbers(numbers)
        self.species = _
        self.sorted_species = __
        self.sorted_numbers = ___
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.loop = loop

        self.NA = len(numbers)
        self.NS = len(self.species)
        self.NO = int((3 ** lmax - 1) / 2)
        self.Oidx = []
        for i in range(lmax):
            self.Oidx.extend([i] * (2*i + 1))

        self.α = -(tc.rand(self.NS, nmax, device=cell.device) + 0.2)
        self.rs = tc.rand(self.NS, nmax, device=cell.device)
        # NS x nmax
        self.species_params = tc.rand(self.NS, self.nmax, device=cell.device)
        # Loop x NO x nmax x NO
        self.orbital_params = tc.rand(self.nmax, self.NO, device=cell.device)[None, None].repeat(
                                      loop, lmax, 1, 1)

        layers = (
                 nn.Linear(self.NO, int(1.2 * self.NO), device=cell.device).double(),
                 nn.Tanh().double(),
                 nn.Linear(int(1.2 * self.NO), int(1.2 * nmax), device=cell.device).double(),
                 nn.Tanh().double(),
                 nn.Linear(int(1.2 * nmax), nmax, device=cell.device).double()
                 )

        moduledict = nn.ModuleDict()
        for spe in self.numbers:
            moduledict[str(spe)] = nn.Sequential(*layers)

        gjkwargs = dict(numbers=self.numbers, nmax=nmax)
        a = [Gj(moduledict, **gjkwargs) for j in range(self.loop)]
        self.gj = nn.ModuleList(a)

    def forward(self, tensor):
        dtype, device = tensor.dtype, tensor.device
        NA, NO, NN = self.NA, self.NO, len(tensor)
        nmax, lmax, rcut = self.nmax, self.lmax, self.rcut
        params = (device, dtype, NA, NO, nmax)
        ans = tc.zeros((NN, NA, NO),
                       dtype=dtype, device=device)
        for n, positions in enumerate(tensor):
            iidx, jidx, disp, dist = get_neighbors_info(pnl=self.pnl, pbc=self.pbc,
                                cell=self.cell, positions=positions)
            # iidx = tc.Tensor(iidx).long()
            jatn = tc.Tensor([self.sorted_numbers[j] for j in jidx]).long().to(device=device)

            # disp = positions[iidx] - positions[jidx] @ shift
            # dist = tc.linalg.norm(disp, axis=2)

            # NN
            fcut = self.cutoff(dist, rcut)
            # NN x nmax
            radial = gauss(dist, jatn, self.α, self.rs)
            # NN x NO
            angle = angular(disp, lmax, NO, dtype, device)
            # NNxNOx1 x NNx1xnmax -> NNxNOxnmax
            Fxyz = (fcut[..., None, None] * angle[..., None] * radial[:, None])
            # Sxnmax -> NA x nmax
            Csn = self.species_params[self.sorted_numbers]
            # Loop x lmax x nmax x O -> O x nmax x O
            Wln = self.orbital_params[0, self.Oidx]
            # NA x O
            ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
            for i in range(1, self.loop):
                # NAxO -> NAxnmax
                Csn = Csn + self.gj[i-1](ρ)
                # Loop x lmax x nmax x O -> O x nmax x O
                Wln = self.orbital_params[i, self.Oidx]
                # NA x O
                ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
            ans[n] = ρ
        return ans

    def cutoff(self, dist, rcut):
        """
        dist: NNTot arr
        rcut: float
        return NNTot arr
        """
        # return tc.ones(dist.shape, dtype=dist.dtype, device=dist.device)
        return 0.25 * (tc.cos(dist * np.pi/rcut)+1)**2

    def primitive_neighbor_list(self, cutoff):
        from taps.utils.neighborlist import PrimitiveNeighborList
        return PrimitiveNeighborList(cutoff, bothways=True,
                                     self_interaction=False)

class REANN2(nn.Module):
    """
    Examples
    --------
    from torch import nn
    from taps.ml.descriptors.torch import REANN
    A = len(init)
    pbc = [True, True, True]
    cutoffs = [6.] * A

    desc = REANN(pbc=pbc, cell=cell, cutoffs=cutoffs, numbers=numbers)
    from ase.neighborlist import PrimitiveNeighborList
        pnl = PrimitiveNeighborList(cutoffs=[6]*len(init),
                                    self_interaction=False, bothways=True)
        pnl.build(pbc, init.cell, init.positions)
    """
    def __init__(self, pnl=None, pbc=None, device=None, cutoffs=None,
                 numbers=None,
                 rcut=6., lmax=2, nmax=2,
                 loop=1, **kwargs):
        super(REANN, self).__init__()

        self.pnl = pnl or self.primitive_neighbor_list(cutoffs)
        self.pbc = tc.tensor(pbc).to(device=device)
        self.device = device
        # self.cutoff = cutoff
        self.numbers = numbers
        _, __, ___ = sort_atomic_numbers(numbers)
        self.species = _
        self.sorted_species = __
        self.sorted_numbers = ___
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.loop = loop

        self.NA = len(numbers)
        self.NS = len(self.species)
        self.NO = int((3 ** lmax - 1) / 2)
        self.Oidx = []
        for i in range(lmax):
            self.Oidx.extend([i] * (2*i + 1))

        self.α = -(tc.rand(self.NS, nmax, device=device) + 0.2)
        self.rs = tc.rand(self.NS, nmax, device=device)
        # NS x nmax
        self.species_params = tc.rand(self.NS, self.nmax, device=device)
        # Loop x NO x nmax x NO
        self.orbital_params = tc.rand(self.nmax, self.NO, device=device)[None, None].repeat(
                                      loop, lmax, 1, 1)

        layers = (
                 nn.Linear(self.NO, int(1.2 * self.NO), device=device).double(),
                 nn.Tanh().double(),
                 nn.Linear(int(1.2 * self.NO), int(1.2 * nmax), device=device).double(),
                 nn.Tanh().double(),
                 nn.Linear(int(1.2 * nmax), nmax, device=device).double()
                 )

        moduledict = nn.ModuleDict()
        for spe in self.numbers:
            moduledict[str(spe)] = nn.Sequential(*layers)

        gjkwargs = dict(numbers=self.numbers, nmax=nmax)
        a = [Gj(moduledict, **gjkwargs) for j in range(self.loop)]
        self.gj = nn.ModuleList(a)

    def forward(self, tensor, cells):
        dtype, device = tensor.dtype, self.device
        NA, NO, NN = self.NA, self.NO, len(tensor)
        nmax, lmax, rcut = self.nmax, self.lmax, self.rcut
        params = (device, dtype, NA, NO, nmax)
        ans = tc.zeros((NN, NA, NO),
                       dtype=dtype, device=device)
        for n, positions in enumerate(tensor):
            cell = cells[n]
            iidx, jidx, disp, dist = get_neighbors_info(pnl=self.pnl, pbc=self.pbc,
                                cell=cell, positions=positions)
            # iidx = tc.Tensor(iidx).long()
            jatn = tc.Tensor([self.sorted_numbers[j] for j in jidx]).long().to(device=device)

            # disp = positions[iidx] - positions[jidx] @ shift
            # dist = tc.linalg.norm(disp, axis=2)

            # NN
            fcut = self.cutoff(dist, rcut)
            # NN x nmax
            radial = gauss(dist, jatn, self.α, self.rs)
            # NN x NO
            angle = angular(disp, lmax, NO, dtype, device)
            # NNxNOx1 x NNx1xnmax -> NNxNOxnmax
            Fxyz = (fcut[..., None, None] * angle[..., None] * radial[:, None])
            # Sxnmax -> NA x nmax
            Csn = self.species_params[self.sorted_numbers]
            # Loop x lmax x nmax x O -> O x nmax x O
            Wln = self.orbital_params[0, self.Oidx]
            # NA x O
            ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
            for i in range(1, self.loop):
                # NAxO -> NAxnmax
                Csn = Csn + self.gj[i-1](ρ)
                # Loop x lmax x nmax x O -> O x nmax x O
                Wln = self.orbital_params[i, self.Oidx]
                # NA x O
                ρ = get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
            ans[n] = ρ
        return ans

    def cutoff(self, dist, rcut):
        """
        dist: NNTot arr
        rcut: float
        return NNTot arr
        """
        # return tc.ones(dist.shape, dtype=dist.dtype, device=dist.device)
        return 0.25 * (tc.cos(dist * np.pi/rcut)+1)**2

    def primitive_neighbor_list(self, cutoff):
        from taps.utils.neighborlist import PrimitiveNeighborList
        return PrimitiveNeighborList(cutoff, bothways=True,
                                     self_interaction=False)



class MLIP:
    None
