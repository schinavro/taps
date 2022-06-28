import torch as tc
from taps.projectors import Projector


class Tensor(Projector):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    @Projector.pipeline
    def x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(True).to(
               device=self.device)

    @Projector.pipeline
    def _x(self, coords):
        return tc.from_numpy(coords).double().requires_grad_(True).to(
               device=self.device)

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
