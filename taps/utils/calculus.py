import numpy as np
import torch as tc


def get_finite_gradients(f, x, eps=1e-2):
    """
    x : NxAx3
    newcoords:
    """
    if isinstance(x, np.ndarray):
        md = np
    elif isinstance(x, tc.Tensor):
        md = tc
    N, A, _ = x.shape
    D = A*_
    shap = (A, _)
    shape = (N, A, _)

    gradients = md.zeros((N, D), dtype=float)
    dx = md.zeros(shape, dtype=float)
    fx = f(x)
    for d in range(D):
        idx = (slice(None), *np.unravel_index(d, shap))
        dx[idx] = eps
        gradients[:, d] = (f(x + dx) - fx)/eps
        dx[idx] = 0.
    return gradients.reshape(N, A, 3)


def get_finite_hessian(f, x, eps=1e-2):
    """
    x : NxAx3
    newcoords:
    """
    if isinstance(x, np.ndarray):
        md = np
    elif isinstance(x, tc.Tensor):
        md = tc
    N, A, _ = x.shape
    D = A*_
    shap = (A, _)
    shape = (N, A, _)
    hessian = md.zeros((N, D, D))
    dx1 = md.zeros(shape)
    dx2 = md.zeros(shape)
    E0 = f(x)
    E = md.zeros((N, D))
    for d1 in range(D):
        idx1 = (slice(None), *np.unravel_index(d1, shap))
        dx1[idx1] = eps
        E[:, d1] = f(x + dx1)
        for d2 in range(d1+1):
            idx2 = (slice(None), *np.unravel_index(d2, shap))
            dx2[idx2] = eps
            E12 = f(x + dx1 + dx2)
            E1 = E[:, d1]
            E2 = E[:, d2]
            hessian[:, d1, d2] = (E12 - E1 - E2 + E0) / eps**2
            hessian[:, d2, d1] = hessian[:, d1, d2]
            dx2[idx2] = 0.
        dx1[idx1] = 0.
    return hessian


def get_finite_hessian_from_gradients(g, x, eps=1e-4):
    if isinstance(x, np.ndarray):
        md = np
    elif isinstance(x, tc.Tensor):
        md = tc

    N, A, _ = x.shape
    D = A*_
    shap = (A, _)
    shape = (N, A, _)

    dx = md.zeros(shape)
    G = md.zeros((*shape, 2*D))

    for d in range(D):
        idx = (slice(None), *np.unravel_index(d, shap))
        dx[idx] = eps

        G[..., d] = g(x + dx)        # NxAx3
        G[..., -(d+1)] = g(x - dx)    # NxAx3
        dx[idx] = 0.

    hessian = md.zeros((N, D, D))
    for d1 in range(D):
        i = (slice(None), *np.unravel_index(d1, shap))
        for d2 in range(D):
            j = (slice(None), *np.unravel_index(d2, shap))

            ij = (*i, d2)
            i_j = (*i, -(d2+1))
            ji = (*j, d1)
            j_i = (*j, -(d1+1))
            hessian[:, d1, d2] = G[ij] - G[i_j] + G[ji] - G[j_i]

    return hessian / (4 * eps)
