import numpy as np
from numpy import newaxis as nax
from numpy import atleast_3d

from scipy.fftpack import idst, dst
# from ase.data import atomic_masses


class Projector:
    """
    Coordinate transformation using projector function
    projecting force requires to return forces and coordinate
    because of double recursion problem

    Parameters
    ----------

    domain: Cartesian class
     `Coord` class object containing information about the coordinates before
     projection
    codomain: Cartesian class
     `Coord` class object containing information about projected coordinates
    pipeline: Projector class
      projector nested in the projector. It will recursively call the projector
    """

    def __init__(self, domain=None, codomain=None, pipeline=None):
        self.domain = domain
        self.codomain = codomain
        self.pipeline = pipeline

    def pipeline(prj):
        """
        """
        name = prj.__name__

        def x(self, coords):
            """
            Get Cartesian class and return Cartesian class
            """
            if self.pipeline is not None:
                coords = getattr(self.pipeline, name)(coords)
            if self.codomain is not None:
                codomain = self.codomain.copy()
            else:
                codomain = coords
            codomain.coords = prj(self, coords.coords)
            return codomain

        def _x(self, coords):
            """
            Get numpy array and return numpy array
            """
            if self.pipeline is not None:
                coords = getattr(self.pipeline, name)(coords)
            return prj(self, coords)

        def x_inv(self, coords):
            if self.domain is not None:
                domain = self.domain.copy()
            else:
                domain = coords
            if self.pipeline is not None:
                domain.coords = prj(self, coords.coords)
                return getattr(self.pipeline, name)(domain)
            domain.coords = prj(self, coords)
            return domain

        def _x_inv(self, coords):
            """
            Get numpy array and return numpy array
            """
            if self.pipeline is not None:
                coords = prj(self, coords)
                return getattr(self.pipeline, name)(coords)
            return prj(self, coords)

        def f(self, forces, coords):
            if self.pipeline is not None:
                forces, coords = getattr(self.pipeline, name)(forces, coords)
            return prj(self, forces, coords)

        def f_inv(self, forces, coords):
            if self.pipeline is not None:
                forces, coords = prj(self, forces, coords)
                return getattr(self.pipeline, name)(forces, coords)
            return prj(self, forces, coords)

        def h(self, hess, coords):
            if self.pipeline is not None:
                hess, coords = getattr(self.pipeline, name)(hess, coords)
            return prj(self, hess, coords)

        def h_inv(self, hess, coords):
            if self.pipeline is not None:
                hess, coords = prj(self, hess, coords)
                return getattr(self.pipeline, name)(hess, coords)
            return prj(self, hess, coords)

        def m(self, masses, coords):
            if self.pipeline is not None:
                masses, coords = getattr(self.pipeline, name)(masses, coords)
            return prj(self, masses, coords)

        def m_inv(self, masses, coords):
            if self.pipeline is not None:
                masses, coords = prj(self, masses, coords)
                return getattr(self.pipeline, name)(masses, coords)
            return prj(self, masses, coords)

        return locals()[name]

    @pipeline
    def x(self, coords):
        return coords

    @pipeline
    def _x(self, coords):
        return coords

    @pipeline
    def x_inv(self, coords):
        return coords

    @pipeline
    def _x_inv(self, coords):
        return coords

    @pipeline
    def f(self, forces, coords):
        return forces, coords

    @pipeline
    def f_inv(self, forces, coords):
        return forces, coords

    @pipeline
    def h(self, hessian, coords):
        return hessian, coords

    @pipeline
    def h_inv(self, hessian, coords):
        return hessian, coords

    @pipeline
    def m(self, masses, coords):
        return masses, coords

    @pipeline
    def m_inv(self, masses, coords):
        return masses, coords

    pipeline = staticmethod(pipeline)


class Mask(Projector):
    """
    Mask projector for ignore some coordinates during optimization or
    model calculation

    Parameters
    ----------

    mask: boolean array
        If it should be considered use true else False
    idx: Int
        the number of unmasked coordinate
    orig_coord: array
        For inverse operation, it will be filled up.
    orig_mass: array
        For inverse operation, it will be filled up.
    """
    def __init__(self, mask=None, orig_coord=None, orig_mass=None, **kwargs):
        self.mask = mask
        self.idx = np.arange(len(mask))[self.mask].reshape(-1)
        self.orig_coord = orig_coord
        self.orig_mass = orig_mass

        super().__init__(**kwargs)

    @Projector.pipeline
    def x(self, coords):
        return coords[..., self.mask, :]

    @Projector.pipeline
    def _x(self, coords):
        return coords[..., self.mask, :]

    @Projector.pipeline
    def x_inv(self, coords):
        N = coords.shape[-1]
        orig_coords = self.orig_coord[..., np.newaxis] * np.ones(N)
        orig_coords[..., self.mask, :] = coords
        return orig_coords

    @Projector.pipeline
    def _x_inv(self, coords):
        N = coords.shape[-1]
        orig_coords = self.orig_coord[..., np.newaxis] * np.ones(N)
        orig_coords[..., self.mask, :] = coords
        return orig_coords

    @Projector.pipeline
    def f(self, forces, coords):
        return forces[..., self.mask, :], coords[..., self.mask, :]

    @Projector.pipeline
    def f_inv(self, forces, coords):
        N = coords.shape[-1]
        orig_forces = np.zeros((*self.orig_coord.shape, N))
        orig_forces[..., self.mask, :] = forces
        orig_coords = (self.orig_coord[..., np.newaxis] * np.ones(N))
        orig_coords[..., self.mask, :] = coords
        return orig_forces, orig_coords

    @Projector.pipeline
    def h(self, hessian, coords):
        hmask = self.get_hmask()

        return hessian[hmask][..., hmask, :], coords[..., self.mask, :]

    @Projector.pipeline
    def h_inv(self, hessian, coords):
        N = coords.shape[-1]
        shape = self.orig_coord.shape
        D = np.prod(shape)
        hmask = self.get_hmask()
        orig_hessian = np.zeros((D, D, N))
        for i, ii in enumerate(np.arange(len(hmask))[hmask]):
            orig_hessian[ii, hmask] = hessian[i]
        orig_coords = (self.orig_coord[..., np.newaxis] * np.ones(N))
        orig_coords[..., self.mask, :] = coords
        return orig_hessian, orig_coords

    @Projector.pipeline
    def m(self, masses, coords):
        return masses[..., self.mask, :], coords[..., self.mask, :]

    @Projector.pipeline
    def m_inv(self, masses, coords):
        N = coords.shape[-1]
        orig_masses = np.zeros((*self.orig_coord.shape, N))
        orig_masses[..., self.mask, :] = masses
        orig_coords = (self.orig_coord[..., np.newaxis] * np.ones(N))
        orig_coords[..., self.mask, :] = coords
        return orig_masses, orig_coords

    def get_hmask(self, mask=None):
        shape = self.orig_coord.shape
        if len(shape) == 1:
            return self.mask
        return (np.array([self.mask] * shape[0])).flatten()


class Sine(Projector):
    """
    Tools for dimensional reducement.

    Sine Projector
    --------------
    init : array; D or 3 x A
    fin  : array; D or 3 x A
    """
    def __init__(self, N=None, Nk=None, init=None, fin=None, **kwargs):
        self.N = N
        self.Nk = Nk
        self.init = init
        self.fin = fin
        self.shape = init.shape
        super().__init__(**kwargs)

    @Projector.pipeline
    def x(self, coords):
        line = self.line()
        # return dst((coords - line), type=1, norm='ortho').flatten()
        return dst((coords - line), type=1, norm='ortho')[..., :self.Nk]

    @Projector.pipeline
    def x_inv(self, rcoords):
        line = self.line()
        # rcoords = rcoords.reshape(*self.shape, self.Nk)
        # return idst(rcoords, type=1, norm='ortho') + line
        _ = np.zeros((*self.shape, self.N-2))
        _[..., :self.Nk] = rcoords.reshape(*self.shape, self.Nk)
        return idst(_, type=1, norm='ortho') + line

    @Projector.pipeline
    def _x(self, coords):
        line = self.line()
        # return dst((coords - line), type=1, norm='ortho').flatten()
        return dst((coords - line), type=1, norm='ortho')[..., :self.Nk]

    @Projector.pipeline
    def _x_inv(self, rcoords):
        line = self.line()
        # rcoords = rcoords.reshape(*self.shape, self.Nk)
        # return idst(rcoords, type=1, norm='ortho') + line
        _ = np.zeros((*self.shape, self.N-2))
        _[..., :self.Nk] = rcoords.reshape(*self.shape, self.Nk)
        return idst(_, type=1, norm='ortho') + line

    @Projector.pipeline
    def f(self, forces, coords):
        """
        forces
        ------
           shape DxNk -> DxN-2

        coords
        ------
           shape DxNk -> DxN-2
        """
        line = self.line()
        coords = dst((coords - line), type=1, norm='ortho')[..., :self.Nk]
        return dst(forces, type=1, norm='ortho')[..., :self.Nk], coords

    @Projector.pipeline
    def f_inv(self, rforces, rcoords):
        line = self.line()
        # rcoords = rcoords.reshape(*self.shape, self.Nk)
        # rforces = rforces.reshape(*self.shape, self.Nk)
        _ = np.zeros((*self.shape, self.N-2))
        __ = np.zeros((*self.shape, self.N-2))
        _[..., :self.Nk] = rcoords.reshape(*self.shape, self.Nk)
        __[..., :self.Nk] = rforces.reshape(*self.shape, self.Nk)
        coords = idst(_, type=1, norm='ortho') + line
        return idst(__, type=1, norm='ortho'), coords

    def line(self):
        if self.__dict__.get('_line') is not None:
            return self._line
        init, fin, N = self.init, self.fin, self.N
        dir = fin - init
        line = np.linspace(0, 1, N)[nax] * dir[..., nax] + init[..., nax]
        self._line = line[..., 1:-1]
        return self._line



class MalonaldehydeProjector(Projector):
    prj_parameters = {
        'reference': {'default': 'None', 'assert': 'True'}}
    mass_type = "variable"
    reference = np.array([[-0.0001, -1.1149, 0.0069],
                          [1.2852, -0.3328, -0.0057],
                          [-1.2852, -0.3328, -0.0056],
                          [0.0000, 1.6434, 0.0000],
                          [-0.0001, -2.2187, 0.0000],
                          [2.2008, -0.9494, -0.0178],
                          [-2.2008, -0.9493, -0.0176],
                          [1.3685, 0.8902, 0.0022],
                          [-1.3685, 0.8902, 0.0022]])

    def __init__(self, reference=None, **kwargs):
        super().prj_parameters.update(self.prj_parameters)
        self.prj_parameters.update(super().prj_parameters)

        self._cache = {'count': 0}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def X(self, positions):
        """
         positions : 3 x N x P
         returns : coords;  D x M x P, in this case 2 x 1 x P
        """
        return positions[:2, 3, np.newaxis, :]

    def X_inv(self, coords):
        """
         coords : D x M x P, in this case 2 x 1 x P
         returns : coords;  3 x N x P
        """
        coords = atleast_3d(coords)
        p = self.reference.T[:, :, np.newaxis] * np.ones(len(coords.T))
        positions = p.copy()  # 3 x N x P
        # Phi, Psi
        positions[:2, 3, np.newaxis] = coords
        return positions

    def F(self, positions, forces):
        """
         positions : 3 x N x P
         forces : 3 x N x P
         return D x M x P; in this case 2 x 1 x P
        """
        return forces[:2, 3, np.newaxis, :]

    def get_effective_mass(self, paths, index=np.s_[1:-1]):
        return np.array([[1]])
