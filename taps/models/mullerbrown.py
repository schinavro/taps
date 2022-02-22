import numpy as np
from taps.models.models import Model


class MullerBrown(Model):
    """ Muller Brown Potential

    .. math::

       \\begin{equation}
       V\\left(x,y\\right) =
       \\sum_{\\mu=1}^{4}{A_\\mu e^{a_\\mu \\left(x-x_\\mu^0\\right)^2
       + b_\\mu \\left(x-x_\\mu^0\\right) \\left(y-y_\\mu^0\\right)
       + c_\\mu\\left(y-y_\\mu^0\\right)^2}}
       \\end{equation}

    * Initial position = (-0.55822365, 1.44172582)

    * Final position = (0.6234994, 0.02803776)

    Parameters
    ----------
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    potential_unit = 'unitless'

    Example
    -------

    >>> import numpy as np
    >>> N = 300
    >>> x = np.linspace(-0.55822365, 0.6234994, N)
    >>> y = np.linspace(1.44172582, 0.02803776, N)
    >>> paths.coords = np.array([x, y])

    """
    implemented_properties = {'potential', 'gradients', 'hessian'}

    A = np.array([-200, -100, -170, 15]) / 100
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    potential_unit = 'unitless'

    def calculate(self, coords, properties=['potential'], **kwargs):
        """
        x : N shape array
        y : N shape array
        return D x N
        """
        if not isinstance(coords, np.ndarray):
            coords = coords.coords
        if len(coords.shape) == 1:
            coords = coords[:, np.newaxis]
        x, y = coords
        A, a, b, c, x0, y0 = self.A, self.a, self.b, self.c, self.x0, self.y0
        x_x0 = (x[:, np.newaxis] - x0)  # N x 4
        y_y0 = (y[:, np.newaxis] - y0)  # N x 4
        Vk = A * np.exp(a * x_x0 ** 2 + b * x_x0 * y_y0 + c * y_y0 ** 2)
        if 'potential' in properties:
            potential = Vk.sum(axis=1)
            self.results['potential'] = potential
        if 'gradients' in properties:
            Fx = (Vk * (2 * a * x_x0 + b * y_y0)).sum(axis=1)
            Fy = (Vk * (b * x_x0 + 2 * c * y_y0)).sum(axis=1)
            self.results['gradients'] = np.array([Fx, Fy])
            # if 'forces' in properties:
            #     self.results['forces'] = -np.array([[Fx], [Fy]])
        if 'hessian' in properties:
            # return 3 x P
            H = np.zeros((2, 1, 2, 1, coords.shape[-1]))
            dx = (2 * a * x_x0 + b * y_y0)
            dy = (b * x_x0 + 2 * c * y_y0)
            Hxx = (Vk * (2 * a + dx * dx)).sum(axis=1)
            Hxy = (Vk * (b + dx * dy)).sum(axis=1)
            Hyy = (Vk * (2 * c + dy * dy)).sum(axis=1)
            H[:, 0, :, 0] = np.array([[Hxx, Hxy], [Hxy, Hyy]])
            self.results['hessian'] = H
