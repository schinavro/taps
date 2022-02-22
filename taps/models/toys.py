import numpy as np
from numpy import newaxis as nax
from taps.models import Model


class Mean(Model):
    implemented_properties = {'potential', 'gradients', 'hessian'}

    def __init__(self, hyperparameters=0., **kwargs):
        self.hyperparameters = hyperparameters
        super().__init__(**kwargs)

    def calculate(self, coords, properties=None, **kwargs):
        shape, N = coords.coords.shape[:-1], coords.N
        D = coords.D

        if 'potential' in properties:
            self.results['potential'] = np.zeros(N) + self.hyperparameters
        if 'gradients' in properties:
            self.results['gradients'] = np.zeros((*shape, N))
        if 'hessian' in properties:
            self.results['hessian'] = np.zeros((D, D, N))

    def __call__(self, coords):
        return self.get_potential(coords=coords)

    def dV(self, coords):
        return self.get_gradients(coords=coords).flatten()

    def ddV(self, coords):
        return self.get_hessian(coords=coords).flatten()

    def set_hyperparameters(self, hyperparameters=None, data=None):
        if hyperparameters is not None:
            self.hyperparameters = hyperparameters

    def get_hyperparameters(self):
        return self.hyperparameters


class Sine(Model):
    """
    TODO: Document it and move it to separate files
    """

    implemented_properties = {'potential', 'gradients', 'hessian'}

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        V = -129.7 + 0.1 * np.sin(4 * coords).sum(axis=(0, 1))
        dV = 0.1 * 4 * np.cos(4 * coords)
        coords = np.atleast_3d(coords)
        D, N = coords.D, coords.N
        _coords = coords.coords.reshape(D, N)
        H = np.zeros((D, N))  # DM x P
        H[0] = -(0.1 * 16) * np.sin(4 * _coords[0])
        H[1] = -(0.1 * 16) * np.sin(4 * _coords[1])
        H = np.einsum('i..., ij->ij...', H, np.identity(D))
        H = H.reshape((D, D, N))

        if np.isscalar(V):
            V = np.array([V])
        self.results['potential'] = V
        self.results['gradients'] = dV
        self.results['hessian'] = H


class Cosine(Model):
    r"""
    TODO: Document it and move it to separate files
    init : -pi -> pi
    fin : -pi -> pi
    $V(x) = V0 + A * \sum_i{cos(\omega * (x + phi))}$
    """
    implemented_properties = {'potential', 'gradients', 'forces', 'hessian'}
    A = -1.
    omega = 4.
    phi = 0.
    V0 = 0.

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        if len(coords.shape) == 1:
            coords = coords[:, nax]
        coords_ = coords.coords + self.phi
        if 'potential' in properties:
            V = self.V0+self.A*np.cos(self.omega*(coords_)).sum(axis=0)
            self.results['potential'] = V
        if 'gradients' in properties or 'forces' in properties:
            F = self.omega * self.A * np.sin(self.omega * (coords_))
            if 'gradients' in properties:
                self.results['gradients'] = -F
            if 'forces' in properties:
                self.results['forces'] = F

        if 'hessian' in properties:
            shape = coords.shape
            omega2A = self.omega * self.omega * self.A
            D = np.prod(shape[:-1])
            N = shape[-1]
            _coords = coords_.reshape(D, N)
            H = np.zeros((D, N))  # D x N
            H = -omega2A*np.cos(self.omega * (_coords+self.phi))
            H = np.einsum('ik, ij->ijk', H, np.identity(D))
            H = H.reshape((D, D, N))
            self.results['hessian'] = H


class Eggholder(Model):
    r"""
    >>> from sympy import symbols, cos, Derivative, summation, IndexedBase
    >>> from sympy import latex
    >>> d, D = symbols('d D', integer=True)
    >>> x = symbols('\mathbf{x}', cls=IndexedBase)
    >>> w, V0, A, d = symbols("\omega V0 A d")
    >>> V = V0 + A * summation(x[d]**2 * cos(w * x[d]), (d, 1, D))
    >>> print(latex(V))
    ... A \sum_{d=1}^{D} \cos{\left(\omega {\mathbf{x}}_{d} \right)}
    ...  {\mathbf{x}}_{d}^{2} + V_{0}
    >>> dV = Derivative(V, x[d], evaluate=True)
    >>> print(latex(dV))
    ... A \sum_{d=1}^{D} \left(- \omega \sin{\left(\omega {\mathbf{x}}_{d}
    ... \right)} {\mathbf{x}}_{d}^{2} + 2 \cos{\left(\omega {\mathbf{x}}_{d}
    ... \right)} {\mathbf{x}}_{d}\right)
    """
    implemented_properties = {'potential', 'gradients', 'forces', 'hessian'}
    A = 1.
    B = 0.1
    omega = 2*np.pi
    phi = 0.
    V0 = 0.

    def __init__(self, A=None, B=None, omega=None, phi=None, V0=None,
                 **kwargs):
        self.A = A or self.A
        self.B = B or self.B

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'], **kwargs):
        if len(coords.shape) == 1:
            coords = coords[:, nax]
        cos = np.cos
        x = coords.coords
        A, B, V0, w, phi = self.A, self.B, self.V0, self.omega, self.phi
        w = self.omega
        if 'potential' in properties:
            V = V0+self.A*(B*x*x - cos(w*(x+phi))).sum(axis=0)
            self.results['potential'] = V

        if 'gradients' in properties:
            dV = A*(2*B*x + w * np.sin(w * (x+phi)))
            self.results['gradients'] = dV

        if 'hessian' in properties:
            shape = coords.shape
            D = np.prod(shape[:-1])
            N = shape[-1]
            x = x.reshape(D, N)
            H = np.zeros((D, N))  # D x N
            H = A * (2*B + w * w * cos(w * (x + phi)))
            H = np.einsum('ik, ij->ijk', H, np.identity(D))
            H = H.reshape((D, D, N))
            self.results['hessian'] = H
