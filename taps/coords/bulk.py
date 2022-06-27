import numpy as np
from taps.coords import Coordinate
from numpy import newaxis as nax
from numpy import concatenate
import numpy

# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
# _iupac2016
atomic_masses = np.array([
    1.0,  # X
    1.008,  # H [1.00784, 1.00811]
    4.002602,  # He
    6.94,  # Li [6.938, 6.997]
    9.0121831,  # Be
    10.81,  # B [10.806, 10.821]
    12.011,  # C [12.0096, 12.0116]
    14.007,  # N [14.00643, 14.00728]
    15.999,  # O [15.99903, 15.99977]
    18.998403163,  # F
    20.1797,  # Ne
    22.98976928,  # Na
    24.305,  # Mg [24.304, 24.307]
    26.9815385,  # Al
    28.085,  # Si [28.084, 28.086]
    30.973761998,  # P
    32.06,  # S [32.059, 32.076]
    35.45,  # Cl [35.446, 35.457]
    39.948,  # Ar
    39.0983,  # K
    40.078,  # Ca
    44.955908,  # Sc
    47.867,  # Ti
    50.9415,  # V
    51.9961,  # Cr
    54.938044,  # Mn
    55.845,  # Fe
    58.933194,  # Co
    58.6934,  # Ni
    63.546,  # Cu
    65.38,  # Zn
    69.723,  # Ga
    72.630,  # Ge
    74.921595,  # As
    78.971,  # Se
    79.904,  # Br [79.901, 79.907]
    83.798,  # Kr
    85.4678,  # Rb
    87.62,  # Sr
    88.90584,  # Y
    91.224,  # Zr
    92.90637,  # Nb
    95.95,  # Mo
    97.90721,  # 98Tc
    101.07,  # Ru
    102.90550,  # Rh
    106.42,  # Pd
    107.8682,  # Ag
    112.414,  # Cd
    114.818,  # In
    118.710,  # Sn
    121.760,  # Sb
    127.60,  # Te
    126.90447,  # I
    131.293,  # Xe
    132.90545196,  # Cs
    137.327,  # Ba
    138.90547,  # La
    140.116,  # Ce
    140.90766,  # Pr
    144.242,  # Nd
    144.91276,  # 145Pm
    150.36,  # Sm
    151.964,  # Eu
    157.25,  # Gd
    158.92535,  # Tb
    162.500,  # Dy
    164.93033,  # Ho
    167.259,  # Er
    168.93422,  # Tm
    173.054,  # Yb
    174.9668,  # Lu
    178.49,  # Hf
    180.94788,  # Ta
    183.84,  # W
    186.207,  # Re
    190.23,  # Os
    192.217,  # Ir
    195.084,  # Pt
    196.966569,  # Au
    200.592,  # Hg
    204.38,  # Tl [204.382, 204.385]
    207.2,  # Pb
    208.98040,  # Bi
    208.98243,  # 209Po
    209.98715,  # 210At
    222.01758,  # 222Rn
    223.01974,  # 223Fr
    226.02541,  # 226Ra
    227.02775,  # 227Ac
    232.0377,  # Th
    231.03588,  # Pa
    238.02891,  # U
    237.04817,  # 237Np
    244.06421,  # 244Pu
    243.06138,  # 243Am
    247.07035,  # 247Cm
    247.07031,  # 247Bk
    251.07959,  # 251Cf
    252.0830,  # 252Es
    257.09511,  # 257Fm
    258.09843,  # 258Md
    259.1010,  # 259No
    262.110,  # 262Lr
    267.122,  # 267Rf
    268.126,  # 268Db
    271.134,  # 271Sg
    270.133,  # 270Bh
    269.1338,  # 269Hs
    278.156,  # 278Mt
    281.165,  # 281Ds
    281.166,  # 281Rg
    285.177,  # 285Cn
    286.182,  # 286Nh
    289.190,  # 289Fl
    289.194,  # 289Mc
    293.204,  # 293Lv
    293.208,  # 293Ts
    294.214,  # 294Og
])


class Bulk(Coordinate):
    """Atomic coordinates for periodic boundary system

    Parameters
    ----------

    number: list of int or ndarray (A, )
      atomic number
    coords: ndarray (N x A x G)
    cell: N x 3 x 3
    """
    implemented_properties = {'masses', 'velocities', 'accelerations'}

    def __init__(self, numbers=None, cell=None, **kwargs):
        self.numbers = numbers
        self.cell = cell
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        return self.coords[idx]

    def __call__(self, index=np.s_[:], coords=None):
        if coords is not None:
            kwargs = self.__dict__.copy()
            del kwargs['coords']
            return self.__class__(coords=coords, **kwargs)
        if index.__class__.__name__ == 'slice' and index == np.s_[:]:
            return self
        kwargs = self.__dict__.copy()
        del kwargs['coords']
        idx = np.arange(self.N)[index].reshape(-1)
        coords = self.coords[idx]
        return self.__class__(coords=coords, **kwargs)

    def __len__(self):
        return len(self.coords)

    def similar(self, sepcies=None, coords=None, cells=None):
        return Bulk(numbers=self.numbers, coords=coords, cell=self.cell)

    def masses(self, **kwargs):
        return atomic_masses[self.numbers][:, nax] * np.ones(self.G)

    def displacements(self, **kwargs):
        """
        Return vector
        """
        init = self.coords[nax, 0]
        # p = coords or self.coords
        return self.coords - init

    def velocities(self, coords=None, index=np.s_[:], **kwargs):
        """ Return velocity at each step
        Get coords and return DxN or 3xAxN array, two point moving average.

        :math:`v[i] = (x[i+1] - x[i]) / dt`

        Parameters
        ----------
        coords : array
            size of DxN or 3xAxN
        epoch : float
            total time step.
        index : slice obj; Default `np.s_[:]`
            Choose the steps want it to be returned. Default is all steps.
        """
        if coords is None:
            p = self.coords.copy()
        else:
            p = coords
        N = self.N
        dt = self.dt
        if index == np.s_[:]:
            p = np.concatenate([p, p[nax, -1]], axis=0)
            return (p[1:] - p[:-1]) / dt
            # vel = np.diff(p, n=1, prepend=p[..., 0, nax]) / dt
            # return vel
        elif index == np.s_[1:-1]:
            return (p[2:] - p[1:-1]) / dt
        i = np.arange(N)[index]
        if i[-1] == N - 1:
            p = concatenate([p, p[nax, -1]], axis=0)
        return (p[i] - p[i - 1]) / dt

    def accelerations(self, coords=None, index=np.s_[:], **kwargs):
        """ Return acceleration at each step
        Get Dx N ndarray, Returns 3xNxP - 1 array, use three point to get
        acceleration

        :math:`a[i] = (2x[i] - x[i+1] - x[i-1]) / dtdt`

        Parameters
        ----------
        coords : array
            size of DxN or 3xAxN
        epoch : float
            total time step.
        index : slice obj; Default `np.s_[:]`
            Choose the steps want it to be returned. Default is all steps.
        """
        if coords is None:
            p = self.coords.copy()
        else:
            p = coords
        N = self.N
        dt = self.dt
        ddt = dt * dt
        if index == np.s_[:]:
            p = concatenate([p[nax, 0], p, p[nax, -1]], axis=0)
            return (2 * p[1:-1] - p[:-2] - p[2:]) / ddt
        elif index == np.s_[1:-1]:
            return (2 * p[1:-1] - p[:-2] - p[2:]) / ddt
        i = np.arange(N)[index]
        if i[0] == 0:
            p = concatenate([p[nax, 0], p], axis=0)
            i += 1
        if i[-1] == N - 1:
            p = concatenate([p, p[nax, -1]], axis=0)
        return (2 * p[i] - p[i - 1] - p[i + 1]) / ddt

    def get_kinetics(self, properties=['kinetic_energies'],
                     return_dict=False, **kwargs):
        """
        Dumb way of calculate.. but why not.
        """
        if type(properties) == str:
            properties = [properties]

        # Make a list of requirments for minimal calulation
        irreplaceable = set()
        for prop in properties:
            if prop in ['masses', 'momentums', 'kinetic_energies',
                        'kinetic_energy_gradients']:
                irreplaceable.add('masses')
            if prop in ['displacements']:
                irreplaceable.add('displacements')
            if prop in ['velocities', 'distances', 'speeds', 'momentums',
                        'kinetic_energies']:
                irreplaceable.add('velocities')
            if prop in ['accelerations', 'kinetic_energy_gradients']:
                irreplaceable.add('accelerations')

        # Calculate
        parsed_properties = list(irreplaceable)
        parsed_results = {}
        for prop in parsed_properties:
            parsed_results[prop] = getattr(self, prop)(**kwargs)

        # Name convention
        m, d, v, a = 'masses', 'displacements', 'velocities', 'accelerations'
        dt = self.dt
        # Assemble
        results = {}
        for prop in properties:
            if prop in [m, d, v, a]:
                results[prop] = parsed_results[prop]
            elif prop == 'epoch':
                results['epoch'] = self.epoch
            elif prop in ['distances', 'speeds', 'kinetic_energies']:
                if results.get(prop) is not None:
                    continue
                vv = parsed_results[v] * parsed_results[v]
                N = len(vv)
                if 'kinetic_energies' in properties:
                    mvv = parsed_results[m] * vv
                    results[prop] = 0.5 * mvv.reshape(N, -1).sum(axis=1)
                else:
                    lvl = np.sqrt(vv.reshape(N, -1).sum(axis=1))
                if 'speeds' in properties:
                    results['speeds'] = lvl
                if 'distances' in properties:
                    results['distances'] = np.add.accumulate(lvl) * dt
            elif prop in ['momentums']:
                results[prop] = parsed_results[m] * parsed_results[v]
            elif prop in ['kinetic_energy_gradients']:
                results[prop] = m * parsed_results[a]

        if len(properties) == 1 and not return_dict:
            return results[properties[0]]
        return results

    def simple_coords(self):
        """Simple line connecting between init and fin"""
        coords = np.zeros(self.coords.shape)
        init = self.coords[[0]]
        fin = self.coords[[-1]]
        dist = fin - init                                # 1xD or 1xAx3
        simple_line = np.linspace(0, 1, self.N)[:, nax, nax] * dist   # N
        coords = (simple_line + init)            # N x A x 3 -> 3 x A x N
        return coords

    def fluctuate(self, initialize=False, cutoff_f=10, fluctuation=0.03,
                  fourier={'type': 1}, seed=None):
        """Give random fluctuation"""
        from scipy.fftpack import idst
        if seed is not None:
            np.random.seed(seed)
        rand = np.random.rand
        NN = np.sqrt(2 * (cutoff_f + 1))
        if initialize:
            self.coords = self.simple_coords()
        size = self.coords[1:-1].shape
        fluc = np.zeros(size)
        fluc[:cutoff_f] = fluctuation * (0.5 - rand(cutoff_f, *size[1:]))
        self.coords[1:-1] += idst(fluc, axis=0, **fourier) / NN

    @property
    def G(self):
        return self.coords.shape[-1]

    @property
    def N(self):
        return len(self.coords)

    @property
    def A(self):
        return self.coords.shape[1]

    @property
    def D(self):
        return np.prod(self.coords.shape[1:])
