import numpy as np
from taps.model.model import Model


class MullerBrown(Model):
    implemented_properties = {'potential', 'gradients', 'hessian', 'forces'}

    model_parameters = {
        'A': {'default': 'np.array([-200, -100, -170, 15])', 'assert': 'True'},
        'a': {'default': 'np.array([-1, -1, -6.5, 0.7])', 'assert': 'True'},
        'b': {'default': 'np.array([0, 0, 11, 0.6])', 'assert': 'True'},
        'c': {'default': 'np.array([-10, -10, -6.5, 0.7])', 'assert': 'True'},
        'x0': {'default': 'np.array([1, 0, -0.5, -1])', 'assert': 'True'},
        'y0': {'default': 'np.array([0, 0.5, 1.5, 1])', 'assert': 'True'},
     }
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    potential_unit = 'unitless'

    def __init__(self, **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)
        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'],
                  **kwargs):
        """
        x : P shape array
        y : P shape array
        """
        self.results = getattr(self, 'results', {})
        x, y = coords.reshape(2, -1)
        A, a, b, c, x0, y0 = self.A, self.a, self.b, self.c, self.x0, self.y0
        x_x0 = (x[:, np.newaxis] - x0)  # P x 4
        y_y0 = (y[:, np.newaxis] - y0)  # P x 4
        Vk = A * np.exp(a * x_x0 ** 2 + b * x_x0 * y_y0 + c * y_y0 ** 2) / 100
        if 'potential' in properties:
            potential = Vk.sum(axis=1)
            self.results['potential'] = potential
        if 'gradients' in properties or 'forces' in properties:
            Fx = (Vk * (2 * a * x_x0 + b * y_y0)).sum(axis=1)
            Fy = (Vk * (b * x_x0 + 2 * c * y_y0)).sum(axis=1)
            if 'gradients' in properties:
                self.results['gradients'] = np.array([[Fx], [Fy]])
            if 'forces' in properties:
                self.results['forces'] = -np.array([[Fx], [Fy]])
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
