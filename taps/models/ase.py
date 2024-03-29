
import copy
import numpy as np
from collections import OrderedDict

from taps.models.models import Model

from ase.atoms import Atoms
from ase.data import atomic_masses
import hashlib
import binascii
from pathlib import Path
from multiprocessing import Pool


# from ase.calculators.calculator import get_calculator_class
# from ase.symbols import Symbols


def get_directory(directory, coord):
    unique_hash = generate_unique_hash(coord)
    return directory + '/' + unique_hash


def generate_unique_hash(positions):
    """return string that explains current calculation
    """
    dk = hashlib.pbkdf2_hmac('sha512', b"Heyy!C@reful!ThisisMyP@ssword!",
                             positions.tobytes(), 1234)
    return str(binascii.hexlify(dk))[2:-1]


class Calculator:
    def __init__(self, image, properties, directory=None):
        self.image = image
        self.properties = properties
        self.directory = directory

    def __call__(self, positions):
        image = self.image
        results = {}
        image.positions = positions
        if self.directory is not None:
            directory = get_directory(self.directory, positions)
            image.calc.directory = directory

        for property in ['potential', 'gradients',
                         'initial_positions', 'positions',
                         'forces', 'hessian']:

            if property not in self.properties:
                continue
            if property == 'potential':
                results[property] = image.get_potential_energy()
            if property in ['gradients', 'forces']:
                forces = image.get_forces()
                results['forces'] = forces
                results['gradients'] = -forces
            if property == 'initial_positions':
                results[property] = image.positions
            if property == 'hessian':
                results[property] = image.get_hessian()

        if 'positions' in self.properties:
            results[property] = image.positions

        return results


class ASE2(Model):
    """

    """
    implemented_properties = ['stresses', 'potential', 'hessian',
                              'gradients', 'positions', 'forces']

    def __init__(self, image=None, directory=None, parallel=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.directory = directory
        self.parallel = parallel

    def calculate(self, coords, properties=['potential'], **kwargs):
        if not isinstance(coords, np.ndarray):
            coords = coords.coords
        if len(coords.shape) == 2:
            coords = coords[np.newaxis, ...]

        results_raw = []
        script = Calculator(self.image, properties, directory=self.directory)

        if self.parallel:
            with Pool(16) as pool:
                results_raw = pool.map(script, coords)
        else:
            for positions in coords:
                results_raw.append(script(positions))

        self.results = self.packaging(results_raw)

    def packaging(self, raw):
        """
        Convert list of dictionaries to dictionary of arrays
        """
        N = len(raw)
        if N == 0:
            return {}

        results = {}
        for key, value in raw[0].items():
            if key == 'potentials':
                results[key] = np.zeros((N, *value.shape))
            elif key == 'potential':
                results[key] = np.zeros(N)
            elif key == 'forces' or key == 'gradients':
                results[key] = np.zeros((N, *value.shape))
            elif key == 'positions' or key == 'initial_positions':
                results[key] = np.zeros((N, *value.shape))
            elif key == 'hessian':
                results[key] = np.zeros((N, *value.shape))
            else:
                results[key] = np.zeros((N, *value.shape))

        for i in range(N):
            for key, value in raw[i].items():
                results[key][i] = value

        return results

    def get_masses(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        return atomic_masses[image.symbols.numbers, np.newaxis]

    def get_effective_mass(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        m = atomic_masses[image.symbols.numbers]
        return np.repeat(m, 3)[np.newaxis]

    @classmethod
    def Coord2image(cls, coord, label, **kwargs):
        image = Atoms(positions=coord, **kwargs)
        image.calc.label = label
        return copy.deepcopy(image)

    def write(self, paths, coords, filename=None, **kwargs):
        """
        Use trajectory to write..
        """
        from ase.io.trajectory import TrajectoryWriter
        if not (".traj" in filename or ".trj" in filename):
            filename = filename + ".traj"
        if not isinstance(coords, np.ndarray):
            coords = coords.coords
        trj = TrajectoryWriter(filename, mode="a")
        N = coords.shape[-1]
        image = self.image.copy()
        for n in range(N):
            positions = coords[:, :, n]
            image.positions = positions
            trj.write(image)


class ASE(Model):
    """

    """
    implemented_properties = ['stresses', 'potential', 'hessian',
                              'gradients', 'positions', 'forces']

    calculation_properties = OrderedDict(
        # stresses='{image}.get_stress()',
        # potentials='{image}.get_potential_energies()',
        potential='{image}.get_potential_energy()',
        # gradients='self.prj.f_inv(-{image}.get_forces().T[..., np.newaxis],'
        #           ' {image}.positions.T[..., np.newaxis])[0][..., 0].T',
        gradients='-{image}.get_forces()',
        initial_positions='positions',
        positions='{image}.positions',
        forces='{image}.get_forces()',
        hessian='self.get_finite_hessian({image}, **kwargs)'
    )

    def __init__(self, image=None, set_label=None, set_directory=None,
                 h_mask=None, **kwargs):

        self.image = image
        self.set_label = set_label
        self.set_directory = set_directory
        self.h_mask = h_mask

        super().__init__(**kwargs)

    def calculate(self, coords, properties=['potential'], **kwargs):
        results_raw = {}
        if not isinstance(coords, np.ndarray):
            coords = coords.coords
        if len(coords.shape) == 2:
            coords = coords[..., np.newaxis]

        for positions in coords.T:
            image = self.positions2image(positions)
            for property in self.calculation_properties.keys():
                if property not in properties:
                    continue
                if results_raw.get(property) is None:
                    results_raw[property] = []
                call = self.calculation_properties[property]
                results_raw[property].append(eval(call.format(image='image')))

        self.results = self.packaging(results_raw)

    def packaging(self, raw):
        results = {}
        for key, value in raw.items():
            if key == 'potentials':
                results[key] = np.array(value).T
            elif key == 'potential':
                results[key] = np.array(value)
            elif key == 'forces' or key == 'gradients':
                results[key] = np.array(value).T
            elif key == 'positions' or key == 'initial_positions':
                results[key] = np.array(value).T
            elif key == 'hessian':
                results[key] = np.array(value).T
            else:
                results[key] = np.array(value).T
        return results

    def get_masses(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        return atomic_masses[image.symbols.numbers, np.newaxis]

    def get_effective_mass(self, paths, coords=None, image=None, **kwargs):
        image = image or self.image
        m = atomic_masses[image.symbols.numbers]
        return np.repeat(m, 3)[..., np.newaxis]

    def get_finite_hessian(self, image, eps=1e-4, h_mask=None, **kwargs):

        mask = h_mask or self.__dict__.get('h_mask', [True] * len(image))
        mmask = np.array([[m]*3 for m in mask]).flatten()
        shape = image.positions.shape
        DD = len(image)*3
        D = len(image[mask]) * 3
        H = np.zeros((DD, DD))
        x = image.positions.copy()     # A x 3
        dx = H + eps * np.identity(DD)  # 3A x 3A

        g = -image.get_forces().flatten()
        g_list = []
        h_index = np.arange(DD)[mmask]

        for ii in range(D):
            i = h_index[ii]
            image.positions = x + dx[i, :].reshape(shape)
            gi = -image.get_forces().flatten()
            g_list.append(gi.copy())
            H[i, i] = (gi[i] - g[i] + gi[i] - g[i]) / (2*eps)
            for jj in range(ii):
                j = h_index[jj]
                gj = g_list[jj]
                H[i, j] = (gj[i] - g[i] + gi[j] - g[j]) / (2*eps)
                H[j, i] = H[i, j]

        return H

    def coord2image(self, coord=None, image=None, set_label=None,
                    set_directory=None):
        image = image or self.image
        set_label = set_label or self.set_label
        set_directory = set_directory or self.set_directory
        image.positions = coord
        if set_label:
            label = self.get_label(coord)
            image.calc.label = label
        if set_directory:
            directory = self.get_directory(coord)
            image.calc.directory = directory

        return copy.deepcopy(image)

    def positions2image(self, positions=None, image=None, set_label=None,
                        set_directory=None):
        image = image or self.image
        set_label = set_label or self.set_label
        set_directory = set_directory or self.set_directory
        image.positions = positions
        coord = positions.T
        if set_label:
            label = self.get_label(coord)
            image.calc.label = label
        if set_directory:
            directory = self.get_directory(coord)
            image.calc.directory = directory

        return copy.deepcopy(image)

    @classmethod
    def Coord2image(cls, coord, label, **kwargs):
        image = Atoms(positions=coord, **kwargs)
        image.calc.label = label
        return copy.deepcopy(image)

    def write(self, paths, coords, filename=None, **kwargs):
        """
        Use trajectory to write..
        """
        from ase.io.trajectory import TrajectoryWriter
        if not (".traj" in filename or ".trj" in filename):
            filename = filename + ".traj"
        if not isinstance(coords, np.ndarray):
            coords = coords.coords
        trj = TrajectoryWriter(filename, mode="a")
        N = coords.shape[-1]
        image = self.image.copy()
        for n in range(N):
            positions = coords[:, :, n].T
            image.positions = positions
            trj.write(image)


class AlanineDipeptide(ASE):

    def get_directory(self, coord=None):
        """
        coord: 3 x A
        """
        directory = self.directory or "."
        prefix = self.prefix or ""

        phi = self.get_dihedral(coord[..., np.newaxis], 4, 6, 8, 14)[0, 0]
        psi = self.get_dihedral(coord[..., np.newaxis], 6, 8, 14, 16)[0, 0]
        ang = ((np.array([phi, psi]) + 180) % 360 - 180)
        # ang[ang > 180.] -= 360.
        _form = 5 + (ang < 0)
        meta = '{0:0{2}.1f}_{1:0{3}.1f}'.format(*ang, *_form)
        name = prefix + meta
        label = directory + '/' + name + '/'
        from pathlib import Path
        Path(label).mkdir(parents=True, exist_ok=True)
        with open(label + 'ICONST', 'w') as f:
            f.write('T 14 21 15 17 0\nT 21 15 17 22 0')
        return label

    def get_dihedral(self, p, a, b, c, d):
        # TODO: Merge at utils.py
        # TODO: Unit add, return form document
        # vector 1->2, 2->3, 3->4 and their normalized cross products:
        # p : 3 x N x P
        v_a = p[:, b, :] - p[:, a, :]  # 3 x P
        v_b = p[:, c, :] - p[:, b, :]
        v_c = p[:, d, :] - p[:, c, :]

        bxa = np.cross(v_b.T, v_a.T).T  # 3 x P
        cxb = np.cross(v_c.T, v_b.T).T
        bxanorm = np.linalg.norm(bxa, axis=0)  # P
        cxbnorm = np.linalg.norm(cxb, axis=0)
        if np.any(bxanorm == 0) or np.any(cxbnorm == 0):
            raise ZeroDivisionError('Undefined dihedral angle')
        bxa /= bxanorm  # 3 x P
        cxb /= cxbnorm
        angle = np.sum(bxa * cxb, axis=0)  # P
        # check for numerical trouble due to finite precision:
        angle[angle < -1] = -1
        angle[angle > 1] = 1
        angle = np.arccos(angle) * 180 / np.pi
        reverse = np.sum(bxa * v_c, axis=0) > 0
        angle[reverse] = 360 - angle[reverse]
        return angle.reshape(1, -1)

    def get_effective_mass(self, paths, index=np.s_[1:-1]):
        model_name = paths.model.__class__.__name__
        masses = atomic_masses[paths._numbers]
        if not model_name == 'Gaussian':
            return np.array([[110], [70]])
        if self.mass_type == 'invariant':
            return np.array([[110], [70]])
        #   return masses
        k, imgdb = paths.model.kernel, paths.imgdb
        hyperparameters = {'sigma_f': 0.1,
                           'sigma_n^f': 1e-3,
                           'sigma_n^e': 1e-4,
                           'l^2': 0.01}
        if self.found_new_data(imgdb):

            for row in imgdb._c.select('id>%d' % self._cache['count']):
                I_phi, I_psi = self.get_inertia(row.positions, masses)
                self._cache['I_phi'].append(I_phi)
                self._cache['I_psi'].append(I_psi)
                self._cache['Xn_T'].append(row.data.X.T)
                self._cache['count'] += 1
            Xn = np.array(self._cache['Xn_T']).T
            self._cache['K_inv'] = np.linalg.inv(
                k(Xn, Xn, orig=True, hyperparameters=hyperparameters))
            self._cache['m_phi'] = np.average(self._cache['I_phi'])
            self._cache['m_psi'] = np.average(self._cache['I_psi'])

        Xm = paths.coords[..., index].copy()
        Xn = np.array(self._cache['Xn_T']).T
        K_s = k(Xn, Xm, orig=True, hyperparameters=hyperparameters)
        K_inv = self._cache['K_inv']
        m_phi = self._cache['m_phi']
        m_psi = self._cache['m_psi']
        Y_phi = np.array(self._cache['I_phi'])
        Y_psi = np.array(self._cache['I_psi'])
        I_phi = m_phi + K_s.T @ K_inv @ (Y_phi - m_phi)
        I_psi = m_psi + K_s.T @ K_inv @ (Y_psi - m_psi)
        return np.array([[I_phi, I_psi]])   # inertia; I_theta, phi
        # return 100

    def prj_f_inv(self, *args, **kwargs):
        return self.prj.f_inv(self.results["gradients"],
                              self.results["positions"])


class AlanineDipeptide3D(ASE):

    def get_labels2(self, paths, idx=None, coords=None):
        if coords is None:
            coords = paths.coords.copy()
        if self.label is None:
            self.label = paths.label
        directory = self.directory or "."
        prefix = self.prefix or ""
        conformation = paths.plotter.conformation
        translation = paths.plotter.translation
        labels = []
        for i in idx:
            # ang = (coords[0, :, i] % (2 * np.pi) + translation) * conformation
            ang = idx
            ang[ang > 180.] -= 360.
            _form = 5 + (ang < 0)
            meta = '{0:0{2}.1f}_{1:0{3}.1f}'.format(*ang, *_form)
            name = prefix + meta
            labels.append(directory + '/' + name + '/' + name)
        return labels
