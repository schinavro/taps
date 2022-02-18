import numpy as np
from numpy import linspace, random
from taps.paths import Paths
from taps.utils.utils import pandas2dct
from ase.ga.utilities import closest_distances_generator
from scipy.spatial import cKDTree as KDTree
from scipy.fftpack import idst


def row2paths(row):
    symbols, coords, dct = row
    return Paths(symbols, coords, **dct)


def read_csv(label, index=0):
    symbols, coords, kwargs = pandas2dct(label=label, index=index)
    return Paths(symbols, coords, **kwargs)


def get_simple_coords(init, fin, N, **pkwargs):
    if getattr(init, 'positions', None) is not None:
        dist = fin.positions - init.positions  # A x 3
        simple_line = linspace(0, 1, N)[:, np.newaxis, np.newaxis] * dist
        coords = (simple_line + init.positions).T  # N x A x 3 -> 3 x A x N
    else:
        dist = fin - init
        simple_line = linspace(0, 1, N)[np.newaxis, :] * dist[..., np.newaxis]
        coords = (simple_line + init[..., np.newaxis])
    return coords


def get_random_coords(init, fin, N, **pkwargs):
    """
    Get initial and final atoms object and returns non-overlapped coords.
    A : int; Number of atoms
    N : int; Number of steps
    mask : list; atoms want to fix. True if it moves.
    """
    A = len(init)
    mask = pkwargs.get('mask', np.array([True] * A))
    mA = mask.sum()
    frequency_number = pkwargs.get('frequency_number', 10)
    fluctuation = pkwargs.get('fluctuation', 5)
    # mic = pkwargs.get('mic', False)

    simple = get_simple_coords(init, fin, N)
    numbers = init.get_atomic_numbers()
    bl = closest_distances_generator(numbers, ratio_of_covalent_radii=0.7)

    # Step between Initial and Final
    fn = frequency_number
    fourier = {'type': 1}
    if len(init) == 1:
        r = np.zeros(3, A, N-2)
        r[..., :fn] = fluctuation * (0.5 - random.rand(3, A, fn))
        simple[..., 1:-1] += idst(r, **fourier) / np.sqrt(2*(fn+1))
        return simple

    trial = 0
    while True:
        trial += 1
        interrupt = False
        r = np.zeros((3, A, N-2))
        r[..., :fn] = fluctuation * (0.5 - random.rand(3, mA, fn))
        masked = simple[:, mask, 1:-1] + idst(r, **fourier) / np.sqrt(2*(fn+1))
        full = simple[:, :, 1:-1].copy()
        full[:, mask, :] = masked
        # if mic:
        #     relpos = np.linalg.solve(cell.T, full.reshape(3, N * (P - 2)))
        #     pb = array([0.5, 0.5, 0.5]) * pbc
        #     mic_full = full + dot(cell, (relpos > pb) - pb).reshape(3, N,
        #                           step)
        #     mic_masked = mic_full[:, mask, :].T
        if trial > 1000:
            # raise PathsNotFoundError('We couldn\'t find the proper path')
            print('We couldn\'t find the proper path')
            simple[:, mask, 1:-1] = masked
            return simple
        full = full.T
        masked = masked.T
        # if np.any(mic):
        #     mic_full = mic_full.T
        #     mic_masked = mic_masked.T
        for a in range(N-2):
            tree = KDTree(full[a])
            for dist, idx in zip(*tree.query(masked[a], k=2)):
                if dist[1] < bl[(numbers[idx[0]], numbers[idx[1]])]:
                    interrupt = True
                    break
            # if not np.any(mic):
            #     continue
            # m_tree = KDTree(mic_full[a])
            # for dist, idx in zip(*m_tree.query(mic_masked[a], k=2)):
            #     if dist[1] < bl[(numbers[idx[0]], numbers[idx[1]])]:
            #         interrupt = True
            #         break
            if interrupt:
                break
        if not interrupt:
            break
    simple[:, mask, 1:-1] = masked.T
    return simple
