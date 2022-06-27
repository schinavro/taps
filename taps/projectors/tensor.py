import torch as tc
from taps.projectors import Projector


class Tensor(Projector):
    @Projector.pipeline
    def x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(True)

    @Projector.pipeline
    def _x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(True)

    def x_inv(self, tensor):
        return tensor.detach().numpy()


class StaticTensor(Projector):
    @Projector.pipeline
    def x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(False)

    @Projector.pipeline
    def _x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(False)

    @Projector.pipeline
    def x_inv(self, tensor):
        return tensor.detach().numpy()
