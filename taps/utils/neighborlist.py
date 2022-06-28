import numpy as np
import torch as tc
from scipy.spatial import KDTree as cKDTree
import itertools
# from ase.geometry import wrap_positions
from ase.geometry import complete_cell
from taps.utils.minkowski_reduction import minkowski_reduce
from ase.cell import Cell


def wrap_positions(positions, cell, pbc=True, center=(0.5, 0.5, 0.5),
                   pretty_translation=False, eps=1e-7):
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.  See also the
    :meth:`ase.Atoms.wrap` method.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: three float
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    pretty_translation: bool
        Translates atoms such that fractional coordinates are minimized.
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    Example:

    >>> from ase.geometry import wrap_positions
    >>> wrap_positions([[-0.1, 1.01, -0.5]],
    ...                [[1, 0, 0], [0, 1, 0], [0, 0, 4]],
    ...                pbc=[1, 1, 0])
    array([[ 0.9 ,  0.01, -0.5 ]])
    """

    if not hasattr(center, '__len__'):
        center = (center,) * 3

    # pbc = pbc2pbc(pbc)
    shift = tc.Tensor(center).long().to(device=cell.device) - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[tc.logical_not(pbc)] = 0.0

    # assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    cell = tc.Tensor(complete_cell(cell.cpu())).double().to(device=cell.device)
    fractional = tc.linalg.solve(cell.T, positions.T).T - shift

    if pretty_translation:
        fractional = translate_pretty(fractional, pbc)
        shift = np.asarray(center) - 0.5
        shift[np.logical_not(pbc)] = 0.0
        fractional += shift
    else:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0
                fractional[:, i] += shift[i]

    return fractional @ cell


def primitive_neighbor_list(quantities, pbc, cell, positions, cutoff,
                            numbers=None, self_interaction=False,
                            use_scaled_positions=False, max_nbins=1e6):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are

            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.

    """

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        empty_types = dict(i=(int, (0, )),
                           j=(int, (0, )),
                           D=(float, (0, 3)),
                           d=(float, (0, )),
                           S=(int, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [tc.Tensor([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = tc.linalg.pinv(cell).T

    # Compute distances of cell faces.
    l1 = tc.linalg.norm(b1_c)
    l2 = tc.linalg.norm(b2_c)
    l3 = tc.linalg.norm(b3_c)
    face_dist_c = tc.Tensor([1 / l1 if l1 > 0 else 1,
                            1 / l2 if l2 > 0 else 1,
                            1 / l3 if l3 > 0 else 1])

    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    else:
        if np.isscalar(cutoff):
            max_cutoff = cutoff
        else:
            cutoff = tc.asarray(cutoff)
            max_cutoff = 2*tc.max(cutoff)

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = tc.maximum((face_dist_c / bin_size).long(), tc.Tensor([1, 1, 1]).long())
    nbins = tc.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = tc.maximum(nbins_c // 2, tc.Tensor([1, 1, 1]).long())
        nbins = tc.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = \
        tc.ceil(bin_size * nbins_c / face_dist_c).long()

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    neigh_search_x = 0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = tc.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = tc.linalg.solve(cell.T,
                                              positions.T).T
    bin_index_ic = tc.floor(scaled_positions_ic*nbins_c).long()
    cell_shift_ic = tc.zeros_like(bin_index_ic).long()

    for c in range(3):
        if pbc[c]:
            # (Note: tc.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = \
                divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = tc.clip(bin_index_ic[:, c], 0, nbins_c[c]-1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = (bin_index_ic[:, 0] +
                   nbins_c[0] * (bin_index_ic[:, 1] +
                                 nbins_c[1] * bin_index_ic[:, 2]))

    # atom_i contains atom index in new sort order.
    atom_i = tc.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = tc.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -tc.ones([nbins, max_natoms_per_bin], dtype=int)
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = tc.cat([tc.Tensor([True]).bool(),
                       bin_index_i[:-1] != bin_index_i[1:]])
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = tc.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    atom_pairs_pn = tc.indices((max_natoms_per_bin, max_natoms_per_bin),
                               dtype=int)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = tc.meshgrid(tc.arange(nbins_c[2]),
                                               tc.arange(nbins_c[1]),
                                               tc.arange(nbins_c[0]),
                                               indexing='ij')
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == tc.arange(tc.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z+1):
        for dy in range(-neigh_search_y, neigh_search_y+1):
            for dx in range(-neigh_search_x, neigh_search_x+1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] *
                              (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                              ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = \
                    atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]

                # Shift vectors.
                _cell_shift_vector_x_n = \
                    tc.resize(shiftx_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftx_xyz.size)).T
                _cell_shift_vector_y_n = \
                    tc.resize(shifty_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shifty_xyz.size)).T
                _cell_shift_vector_z_n = \
                    tc.resize(shiftz_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftz_xyz.size)).T

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = tc.logical_and(_first_at_neightuple_n != -1,
                                      _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = tc.concatenate(first_at_neightuple_nn)
    secnd_at_neightuple_n = tc.concatenate(secnd_at_neightuple_nn)
    cell_shift_vector_n = tc.transpose([tc.concatenate(cell_shift_vector_x_n),
                                        tc.concatenate(cell_shift_vector_y_n),
                                        tc.concatenate(cell_shift_vector_z_n)])

    # Add global cell shift to shift vectors
    cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - \
        cell_shift_ic[secnd_at_neightuple_n]

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = tc.logical_not(tc.logical_and(
            first_at_neightuple_n == secnd_at_neightuple_n,
            (cell_shift_vector_n == 0).all(axis=1)))
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = tc.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = positions[secnd_at_neightuple_n] - \
        positions[first_at_neightuple_n] + \
        cell_shift_vector_n.dot(cell)
    abs_distance_vector_n = \
        tc.sqrt(tc.sum(distance_vector_nc*distance_vector_nc, axis=1))

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    if isinstance(cutoff, dict) and numbers is not None:
        # If cutoff is a dictionary, then the cutoff radii are specified per
        # element pair. We now have a list up to maximum cutoff.
        per_pair_cutoff_n = tc.zeros_like(abs_distance_vector_n)
        for (atomic_number1, atomic_number2), c in cutoff.items():
            try:
                atomic_number1 = atomic_numbers[atomic_number1]
            except KeyError:
                pass
            try:
                atomic_number2 = atomic_numbers[atomic_number2]
            except KeyError:
                pass
            if atomic_number1 == atomic_number2:
                mask = tc.logical_and(
                    numbers[first_at_neightuple_n] == atomic_number1,
                    numbers[secnd_at_neightuple_n] == atomic_number2)
            else:
                mask = tc.logical_or(
                    tc.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number1,
                        numbers[secnd_at_neightuple_n] == atomic_number2),
                    tc.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number2,
                        numbers[secnd_at_neightuple_n] == atomic_number1))
            per_pair_cutoff_n[mask] = c
        mask = abs_distance_vector_n < per_pair_cutoff_n
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    elif not np.isscalar(cutoff):
        # If cutoff is neither a dictionary nor a scalar, then we assume it is
        # a list or numpy array that contains atomic radii. Atoms are neighbors
        # if their radii overlap.
        mask = abs_distance_vector_n < \
            cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n]
        elif q == 'j':
            retvals += [secnd_at_neightuple_n]
        elif q == 'D':
            retvals += [distance_vector_nc]
        elif q == 'd':
            retvals += [abs_distance_vector_n]
        elif q == 'S':
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError('Unsupported quantity specified.')
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)


class PrimitiveNeighborList:
    """Neighbor list that works without Atoms objects.

    This is less fancy, but can be used to avoid conversions between
    scaled and non-scaled coordinates which may affect cell offsets
    through rounding errors.
    """
    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, use_scaled_positions=False):
        self.device = 'cuda' if tc.cuda.is_available() else 'cpu'
        self.cutoffs = tc.Tensor(cutoffs).long().to(device=self.device) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update2(self, pbc, cell, coordinates):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(pbc, cell, coordinates)
            return True
        if ((self.pbc != pbc).any() or (self.cell != cell).any() or
            ((self.coordinates - coordinates)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, coordinates)
            return True

        return False

    def update(self, pbc, cell, coordinates):
        """Make sure the list is up to date."""

        self.build(pbc, cell, coordinates)
        return True

    def build(self, pbc, cell, coordinates):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc
        self.cell = cell
        self.coordinates = coordinates

        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {0} != {1}'
                             .format(len(self.cutoffs), len(coordinates)))

        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        if self.use_scaled_positions:
            # positions0 = cell.cartesian_positions(coordinates)
            positions0 = tc.linalg.solve(cell.T, (coordinates).T).T
        else:
            positions0 = coordinates

        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)

        natoms = len(positions)
        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [tc.empty(0).long().to(device=self.device) for a in range(natoms)]
        self.displacements = [tc.empty(0, 3).long().to(device=self.device) for a in range(natoms)]
        self.nupdates += 1
        if natoms == 0:
            return

        N = []
        ircell = tc.linalg.pinv(rcell)
        for i in range(3):
            if self.pbc[i]:
                v = ircell[:, i]
                h = 1 / tc.linalg.norm(v)
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        tree = cKDTree(positions.detach().cpu().numpy(), copy_data=True)
        # offsets = cell.scaled_positions(positions - positions0)
        offsets = tc.linalg.solve(cell.T, (positions-positions0).T).T
        offsets = offsets.round().long()

        for n1, n2, n3 in itertools.product(range(0, N[0] + 1),
                                            range(-N[1], N[1] + 1),
                                            range(-N[2], N[2] + 1)):
            if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                continue

            displacement = tc.Tensor([n1, n2, n3]).double().to(device=cell.device) @ rcell
            for a in range(natoms):

                pos = positions[a] - displacement
                indices = tree.query_ball_point(pos.detach().cpu().numpy(),
                                                r=(self.cutoffs[a] + rcmax).detach().cpu().numpy())
                if not len(indices):
                    continue

                indices = tc.Tensor(indices).long().to(device=cell.device)
                delta = positions[indices] + displacement - positions[a]
                cutoffs = self.cutoffs[indices] + self.cutoffs[a]
                i = indices[tc.linalg.norm(delta, axis=1) < cutoffs]
                if n1 == 0 and n2 == 0 and n3 == 0:
                    if self.self_interaction:
                        i = i[i >= a]
                    else:
                        i = i[i > a]

                self.nneighbors += len(i)
                self.neighbors[a] = tc.cat([self.neighbors[a], i])

                n123 = tc.Tensor([n1, n2, n3]).long().to(device=cell.device)
                disp = n123.double() @ op.double() + offsets[i] - offsets[a]
                self.npbcneighbors += disp.any(1).sum()
                self.displacements[a] = tc.cat([self.displacements[a], disp])

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                if len(neighbors2[a]) == 0:
                    continue
                nbs = tc.cat([self.neighbors[a], tc.Tensor(neighbors2[a]).long().to(device=self.device)])
                disp = tc.cat([self.displacements[a], tc.vstack(displacements2[a])])
                # Force correct type and shape for case of no neighbors:
                self.neighbors[a] = nbs.long()
                self.displacements[a] = disp.long().reshape((-1, 3))

        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = (i < a)
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = tc.cat([self.neighbors[b], [a]])
                        self.displacements[b] = tc.cat([self.displacements[b], [-offset]])
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this::

          indices, offsets = nl.get_neighbors(42)
          for i, offset in zip(indices, offsets):
              print(atoms.positions[i] + offset @ atoms.get_cell())

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return self.neighbors[a], self.displacements[a]
