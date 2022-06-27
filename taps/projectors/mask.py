import numpy as np
from .projector import Projector


class InverseMask(Projector):
    """
    Parameters
    ----------
    reference: ndarray; Original coordinate Ax3
    mask: array of size A; False if it should be masked when it projected

    """
    def __init__(self, reference, mask, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference
        self.mask = mask

    @Projector.pipeline
    def x(self, coords):
        """
        Get the Nxax3 and returns NxAx3 array
        Parameters
        ----------
        coords: Nxax3 ndarray

        return: NxAx3 ndarray
        """
        N = len(coords)
        pcoords = np.repeat(self.reference[None], N, axis=0)
        pcoords[:, self.mask] = coords
        return pcoords

    @Projector.pipeline
    def x_inv(self, coords):
        return coords[:, self.mask]

    @Projector.pipeline
    def f(self, gradients, coords):
        N = len(coords)
        pcoords = np.repeat(self.reference[None], N, axis=0)
        pcoords[:, self.mask] = coords
        pgrad = np.zeros((N, *self.reference.shape))
        pgrad[:, self.mask] = gradients
        return pgrad, pcoords

    @Projector.pipeline
    def f_inv(self, gradients, coords):
        return gradients[:, self.mask], coords[:, self.mask]

    
