import time
import itertools
import torch as tc
from torch import nn
from torch.nn.parameter import Parameter

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


def compress_symbols(symbols):
    """Make each unqiue atomic symbols into consecutive numbers
     This makes coding easier by creating dense matrix.

    Example
    -------

    >>> compress_symbols([1, 1, 1, 1, 6])
    ... ({1:0, 6:1}, {0: 1, 1: 6}, [0, 1])

    Paramers
    --------
     symbols: List

    Return
    ------
     decompressor: Dict{Int, Int}
     sorted_symbols: List{Int}

    """
    species = list(set(symbols))
    encoder = dict([(spe, i) for i, spe in enumerate(species)])
    decoder = dict([(i, spe) for i, spe in enumerate(species)])
    srtd_n = [encoder[n] for n in symbols]
    return encoder, decoder, srtd_n


def get_nn(symbols, positions, cell, pbc, cutoff=6., device='cpu'):
    """From pbc, symbols, cell, positions info, returns

    Parameters
    ----------
     symbols: NTA Tensor{list of symbols
     positions: NTAx3 Tensor{Double}
     cell: 3x3 Tensor{Double}
     pbc: 3 Tensor{Bool}
         Periodic boundary condition
     cutoff:  Tensor{Double} or float

    Returns
    -------
    nidxs, iidxs, jidxs, disps, dists

    """
    A = len(positions)
    icell = tc.linalg.pinv(cell)
    grid = tc.zeros(3).long().to(device=device)
    grid[pbc] = ((2 * cutoff * tc.linalg.norm(icell, axis=0)).long() + 1)[pbc]

    iidxs, jidxs, disps, dists, isymb, jsymb = [], [], [], [], [], []
    for n1, n2, n3 in itertools.product(range(0, grid[0] + 1),
                                        range(-grid[1], grid[1] + 1),
                                        range(-grid[2], grid[2] + 1)):
        # Skip symmetric displacement

        if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):

            continue

        # Calculate the cell jumping
        jumpidx = tc.Tensor([n1, n2, n3]).double().to(device=device)
        # 3
        displacement = jumpidx @ cell

        # Brute force searchingb

        # Ax3 - 3
        jpositions = positions + displacement
        kpositions = positions - displacement
        # Ax1x3 - 1xAx3 -> AxAx3
        disp1 = jpositions[:, None] - positions[None]
        disp2 = kpositions[:, None] - positions[None]
        # AxA
        dist1 = tc.linalg.norm(disp1, axis=2)
        dist2 = tc.linalg.norm(disp2, axis=2)
        # AxA
        if n1 == 0 and n2 == 0 and n3 == 0:
            mask1 = (dist1 < cutoff) * (dist1 > 1e-8)
            mask2 = (dist2 < cutoff) * (dist2 > 1e-8)
        else:
            mask1 = dist1 < cutoff
            mask2 = dist2 < cutoff
        # Get all True idx
        iidx1, jidx1 = mask1.nonzero(as_tuple=True)
        iidx2, jidx2 = mask2.nonzero(as_tuple=True)
        # Appending
        iidxs.append(iidx1)
        jidxs.append(jidx1)
        isymb.append(symbols[iidx1])
        jsymb.append(symbols[jidx1])
        disps.append(disp1[mask1])
        dists.append(dist1[mask1])

        # Symmetric side appending
        iidxs.append(iidx2)
        jidxs.append(jidx2)
        isymb.append(symbols[iidx2])
        jsymb.append(symbols[jidx2])
        disps.append(disp2[mask2])
        dists.append(dist2[mask2])

    return tc.cat(iidxs), tc.cat(jidxs), tc.cat(isymb), tc.cat(jsymb), tc.cat(disps), tc.cat(dists)


def get_neighbors_info(symbols, positions, cells, crystalidx, pbcs, cutoff=None):
    """
    Parameters
    ----------
        symbols: NTA tensor
          Element of atoms corresponds to positions
        positions: NTAx3 tensor
          Atomic positions
        cells: NCx3x3 tensor
          Cell of each crystal
        crystalidx: NTA
          Crystal index corresponds to positions
        pbcs: NCx3 tensor
          Periodic boundary condition of each crystal

    Returns
    -------
     iidx: NN tensor{Int}
     jidx: NN tensor{Int}
     isym: NN tensor{Int}
     jsym: NN tensor{Int}
     disp: NNx3 Tensor{Double}
     disp: NN Tensor{Double}
    """

    cryset = tc.unique(crystalidx)
    totalidx = tc.arange(len(symbols))

    iidx, jidx, isym, jsym, cidx, disp, dist = [], [], [], [], [], [], []
    for c, cidx in enumerate(cryset):
        cmask = crystalidx == cidx
        position = positions[cmask]
        symbol = symbols[cmask]
        crystali = totalidx[cmask]
        pbc, cell = pbcs[c], cells[c]
        # NN, NN, NNx3, NN
        idx, jdx, isy, jsy, dsp, dst = get_nn(symbol, position, cell, pbc)
        iidx.append(crystali[idx])
        # iidx.append(idx)
        isym.append(isy)
        jsym.append(jsy)

        jidx.append(crystali[jdx])
        # jidx.append(jdx)
        disp.append(dsp)
        dist.append(dst)

    return tc.cat(iidx), tc.cat(jidx), tc.cat(isym), tc.cat(jsym), tc.cat(disp), tc.cat(dist)


class Gj(nn.Module):
    def __init__(self, moduledict, species=None, nmax=None):
        super(Gj, self).__init__()

        self.moduledict = moduledict
        self.species = species
        self.nmax = nmax

    def forward(self, ρ, symbols):
        """  Returns the

        Parameters
        ----------

        ρ : NTA x O
        symbols : NTA number index

        Return
        ------
        NTA x nmax
        """
        NTA, nmax = len(ρ), self.nmax
        coeff = tc.zeros((NTA, nmax), dtype=ρ.dtype, device=ρ.device)

        for spe in self.species:
            mask = spe == symbols
            coeff[mask] = self.moduledict[str(spe)](ρ[mask])
        return coeff


class REANN(nn.Module):
    """ Recursive Embedding Atomic Neural Network

    Parameters
    ----------

    symbols: List
      type of elements

    """
    def __init__(self, species=None, rcut=6., lmax=2, nmax=2, loop=1, device='cpu',
                 **kwargs):
        super(REANN, self).__init__()

        self.device = device
        self.species = species
        self.nmax = nmax
        self.lmax = lmax
        # self.loop = loop
        self.register_buffer('loop', tc.Tensor([loop]).long())
        self.register_buffer('rcut', tc.Tensor([rcut]))

        assert len(species) == max(species) + 1, "Use compressed expression"
        NS = len(species)
        NO = int((3 ** lmax - 1) / 2)
        Oidx = []
        for i in range(lmax):
            Oidx.extend([i] * (2*i + 1))

        self.NS = NS
        self.NO = NO
        self.Oidx = Oidx

        self.α = Parameter(-(tc.rand(NS, nmax, device=device) + 0.2))
        self.rs = Parameter(tc.rand(NS, nmax, device=device))
        # NS x nmax
        self.species_params = Parameter(tc.rand(NS, nmax).to(device=device))
        # Loop x NO x nmax x NO
        self.orbital_params = Parameter(tc.rand(nmax, NO)[None, None].repeat(
                                  loop, lmax, 1, 1).to(device=device))

        layers = (
                 nn.Linear(NO, int(1.2 * NO)),
                 nn.SiLU(),
                 nn.Linear(int(1.2 * NO), int(1.2 * nmax)),
                 nn.SiLU(),
                 nn.Linear(int(1.2 * nmax), nmax)
                 )

        moduledict = nn.ModuleDict().to(device=device)
        for spe in species:
            moduledict[str(spe)] = nn.Sequential(*layers).double()

        gjkwargs = dict(species=species, nmax=nmax)
        a = [Gj(moduledict, **gjkwargs) for j in range(loop)]
        self.gj = nn.ModuleList(a).to(device=device)

    def forward(self, symbols, positions, cells, crystalidx, pbcs):
        """
        Parameters
        ----------
        symbols: NTA tensor
          Element of atoms corresponds to positions
        positions: NTAx3 tensor
          Atomic positions
        cells: NCx3x3 tensor
          Cell of each crystal
        crystalidx: NTA
          Crystal index corresponds to positions
        pbcs: NCx3 tensor
          Periodic boundary condition of each crystal

        Returns
        -------
        NTAx NO Tensor{Double}
            density ρ
        """

        # Number of crystals, Number of total atoms
        iidx, jidx, isym, jsym, disp, dist = \
            get_neighbors_info(symbols, positions, cells, crystalidx, pbcs)

        NTA = len(positions)
        dtype, device = positions.dtype, positions.device
        NO, Oidx = self.NO, self.Oidx
        nmax, lmax, rcut = self.nmax, self.lmax, self.rcut

        # NN -> NN
        fcut = self.cutoff_function(dist, rcut)
        # NN -> NN
        radial = self.gauss(dist, jsym, self.α, self.rs)
        # NN x NO
        angle = self.angular(disp, lmax, NO)
        # NNxNOx1 x NNx1xnmax -> NNxNOxnmax
        Fxyz = (fcut[..., None, None] * angle[..., None] * radial[:, None])

        # NN number of total neighbors
        # NSxnmax -> NTA x nmax
        Csn = self.species_params[symbols]
        # Loop x lmax x nmax x O -> O x nmax x O
        Wln = self.orbital_params[0, Oidx]

        params = (device, dtype, NTA, NO, nmax)
        # NTA x O
        ρ = self.get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
        for i in range(1, self.loop):
            # NTAxO -> NTAxnmax
            Csn = Csn + self.gj[i-1](ρ, symbols)
            # Loop x lmax x nmax x O -> O x nmax x O
            Wln = self.orbital_params[i, Oidx]
            # NTA x O
            ρ = self.get_density(Wln, Csn, Fxyz, iidx, jidx, *params)
        return ρ

    def cutoff_function(self, dist, rcut):
        """
        Parameters
        ----------
        dist: NTA
        rcut: float

        Return
        ------
        NTA arr
        """
        # return tc.ones(dist.shape, dtype=dist.dtype, device=dist.device)
        return 0.25 * (tc.cos(dist * tc.pi/rcut)+1)**2

    def gauss(self, dist, jatn, alpha, rs):
        """
        dist : NN
        sig : NS x nmax
        rs : NS x nmax
        return : NN x nmax
        """

        """
        NN = len(dist)
        dtype, device = dist.dtype, dist.device
        gauss = tc.empty((NN, nmax), dtype=dtype, device=device)
        for i, spe in enumerate(species):
            # nmax
            Sig, Rs = sigma[i], rs[i]
            mask = (symbols == spe)
            # NN x nmax
            tmp = np.exp(Sig * (dist[mask, None] - Rs)**2)
            # NN x 1
            gauss.masked_scatter_(mask[:, None], tmp)
        return gauss
        """
        # NSxnmax -> NNxnmax
        Rs = rs[jatn]
        alp = alpha[jatn]
        # NN x nmax - NN x 1 -> NN x nmax
        return tc.exp(alp * (dist[:, None] - Rs)**2)

    def angular(self, disp, lmax, NO):
        """
        unit : NTA x 3
        disp : NTA x 3
        fcut : NTA
        lmax : int
        return : NTA x O
           where NTA is number of total atoms
                  O is number of orbitals

        Only 1, 2 support.
        """
        NN, dtype, device = len(disp), disp.dtype, disp.device

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

    def get_density(self, Wln, Csn, Fxyz, iidx, jidx, device, dtype, NTA, NO, nmax):
        """
        Parameters
        ----------

        Wln: O x nmax x O
        Csn: NTA x nmax
        Fxyz: NNxNOxnmax

        Returns
        -------
         NTA x O
        """
        # NTAxnmax -> NNxnmax
        cj = Csn[jidx]
        # NNx1xnmax x NNxNOxnmax -> NNxNOxnmax
        cjFxyz = cj[:, None] * Fxyz

        # NN x NO x nmax -> NA x NO x nmax
        bnl = tc.zeros((NTA, NO, nmax), device=device, dtype=dtype).index_add(
                                                0, iidx, cjFxyz)

        # NOx(nmax)xNO * NAxNOx(nmax)x1, -> NA x NO x nmax x NO -> NAxOxO -> NAxO
        return tc.sum(tc.sum(Wln * bnl[..., None], axis=2) ** 2, axis=1)
