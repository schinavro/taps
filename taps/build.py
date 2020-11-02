import copy
import numpy as np
from numpy import array, outer, linspace, random, dot
from ase.atoms import Atoms
from ase.io.trajectory import Trajectory
from ase.pathway.paths import Paths, allowed_properties, necessary_parameters
from ase.pathway.utils import pandas2dct
from ase.ga.utilities import closest_distances_generator
from scipy.spatial import cKDTree as KDTree
from scipy.fftpack import dst, idst


def row2paths(row):
    symbols, coords, dct = row
    return Paths(symbols, coords, **dct)


def read_csv(label, index=0):
    symbols, coords, kwargs = pandas2dct(label=label, index=index)
    return Paths(symbols, coords, **kwargs)


def traj2coords(traj):
    if isinstance(traj, str):
        traj = Trajectory(traj)
    init = traj[0]
    P = len(traj)
    N = len(init)
    symbols = init.symbols
    coords = np.zeros((3, N, P))

    # Construct parameters, which contained at intial atoms; atoms.info
    args = {}
    for key, value in necessary_parameters.items():
        default = eval(value['default'].format(P=P, N=N))
        if key in init.info and np.all(init.info[key] != default):
            args[key] = init.info[key]

    # listing property that will be contained
    contained_property = []
    for key in allowed_properties.keys():
        none = allowed_properties[key]['isNone'].format(atoms='init')
        if not eval(none):
            contained_property.append(key)
            args[key] = []

    # Construct properties
    for i, atoms in enumerate(traj):
        coords[:, :, i] = atoms.positions.T
        for key in contained_property:
            value = allowed_properties[key]['call'].format(atoms='atoms')
            args[key].append(value)

    # Compress the property; if every component is same, it will assume
    # as invariant and squash it into one component
    for key in contained_property:
        value = args[key][0]
        if all(x == value for x in args[key]):
            args[key] = value
    return Paths(symbols, coords, **args)


def paths2traj(paths):
    label = paths.label
    images = copy.deepcopy(paths.images[:])
    P = paths.P
    N = paths.N
    for key, value in necessary_parameters.items():
        default = eval(value['default'].format(P=P, N=N))
        data = paths.__dict__.get(key, None)
        if np.all(data == default) or data is None:
            continue
        images[0].info[key] = paths.__dict__.get(key, None)
    traj = Trajectory(label + '.traj', mode='w')
    for atoms in images:
        traj.write(atoms)
    return Trajectory(label + '.traj', mode='r')


def init2coords(init, fin=None, P=50, setting='simple', variables=[],
                mic=False, distribution=None, fluctuation=1, **kwargs):
    """
    init : Atoms ,
    fin : Atoms | if None, fin = init
    P :

    invariant :
    variables :

    return Paths
    """
    if fin is None:
        fin = init.copy()

    assert isinstance(init, Atoms)
    assert P > 2
    assert setting in ['simple', 'random']
    assert isinstance(mic, bool)
    assert np.isscalar(fluctuation)
    assert np.all(init.get_atomic_numbers() == fin.get_atomic_numbers())

    # Construct parameters, which contained at intial atoms; atoms.info
    args = {}
    for key, value in necessary_parameters.items():
        default = eval(value['default'].format(P=P, N=len(init)))
        if key in init.info and np.all(init.info[key] != default):
            args[key] = init.info[key]
    # Check property contained in init atoms. If you don't specify
    # contained_property as variables, it will autometically
    # assume it as invariant

    for key, value in allowed_properties.items():
        none = value['isNone'].format(atoms='init')
        call = value['call'].format(atoms='init')
        if not eval(none):
            if key in variables:
                args[key] = [eval(call)] * P
            else:
                args[key] = eval(call)

    # Override args with kwargs
    for key, value in kwargs.items():
        args[key] = value

    coords_generator = globals()['get_' + setting + '_coords']
    pkwargs = {
        'mask': args.get('mask', np.array([True] * len(init))), 'mic': mic,
        'distribution': linspace(2, 0, P - 2), 'fluctuation': fluctuation,
        'D': args.get('D', 3)
    }

    symbols = init.symbols
    coords = coords_generator(init, fin, P, **pkwargs)
    return Paths(symbols, coords, **args)


def get_simple_coords(init, fin, P, **pkwargs):
    N = len(init)
    dist = fin.positions - init.positions
    simple_line = outer(linspace(0, 1, P), dist).reshape(P, N, 3)
    coords = (simple_line + init.positions).T
    return coords


def get_random_coords(init, fin, P, **pkwargs):
    mask = pkwargs.get('mask')
    D, N, M, step = pkwargs.get('D'), len(init), mask.sum(), P - 2
    distribution = pkwargs.get('distribution')
    fluctuation = pkwargs.get('fluctuation')
    mic = pkwargs.get('mic')

    cell = init.cell
    pbc = init.pbc
    simple = get_simple_paths(init, fin, P)
    numbers = init.get_atomic_numbers()
    bl = closest_distances_generator(numbers, ratio_of_covalent_radii=0.7)

    # Step between Initial and Final
    fourier = {'type': 1, 'norm': 'ortho'}
    if len(init) == 1:
        r = fluctuation * (0.5 - random.rand(3, 1, step)) * distribution
        masked = idst(dst(simple[:, :, 1:-1], **fourier) + r, **fourier)
        simple[:D, :, 1:-1] = masked[:D]
        return simple

    trial = 0
    while True:
        trial += 1
        interrupt = False
        r = fluctuation * (0.5 - random.rand(3, M, step)) * distribution
        full = simple[:, :, 1:-1].copy()
        masked = idst(dst(full[:, mask, :], **fourier) + r, **fourier)
        full[:, mask, :] = masked
        if mic:
            relpos = np.linalg.solve(cell.T, full.reshape(3, N * (P - 2)))
            pb = array([0.5, 0.5, 0.5]) * pbc
            mic_full = full + dot(cell, (relpos > pb) - pb).reshape(3, N, step)
            mic_masked = mic_full[:, mask, :].T
        if trial > 1000:
            # raise PathsNotFoundError('We couldn\'t find the proper path')
            simple[:, mask, 1:-1] = masked
            return simple
        full = full.T
        masked = masked.T
        if np.any(mic):
            mic_full = mic_full.T
            mic_masked = mic_masked.T
        for p in range(step):
            tree = KDTree(full[p])
            for dist, idx in zip(*tree.query(masked[p], k=2)):
                if dist[1] < bl[(numbers[idx[0]], numbers[idx[1]])]:
                    interrupt = True
                    break
            if not np.any(mic):
                continue
            m_tree = KDTree(mic_full[p])
            for dist, idx in zip(*m_tree.query(mic_masked[p], k=2)):
                if dist[1] < bl[(numbers[idx[0]], numbers[idx[1]])]:
                    interrupt = True
                    break
            if interrupt:
                break
        if not interrupt:
            break
    simple[:, mask, 1:-1] = masked.T
    return simple
