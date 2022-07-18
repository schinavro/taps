import time
import torch
import torch as tc
import numpy as np
from torch import nn
# from torch.autograd import grad
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad
from taps.models import Model
from taps.coords import Coordinate
from taps.utils.calculus import get_finite_hessian, get_finite_gradients


class ImgDataset(tc.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.database.count(table_name='image')

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        dat = self.database.read(ids=[idx+1],
                                 entries={
                                    'coord': 'blob',
                                    'potential': 'blob'})[0]
        return tc.from_numpy(dat['coord'].copy()).double(), dat['potential'][0]


class ImageDatabaseDataset(tc.utils.data.Dataset):

    def __init__(self, database, ids=None):
        entries = dict(coord='blob', pcoord='blob',
                       potential='blob', gradients='blob', pgradients='blob')
        if ids is None:
            data = database.read_all(entries=entries)
        else:
            data = database.read(ids=ids, entries=entries)
        N = len(data)
        shap = data[0]['coord'].shape
        pshap = data[0]['pcoord'].shape
        self.x = tc.empty(N, *shap).double()
        self.px = tc.empty(N, *pshap).double()
        self.y = tc.empty(N).double()
        self.dy = tc.empty(N, *shap).double()
        self.pdy = tc.empty(N, *pshap).double()
        for n in range(N):
            self.x[n] = tc.from_numpy(data[n]['coord'].copy()).double()
            self.px[n] = tc.from_numpy(data[n]['pcoord'].copy()).double()
            self.y[n] = data[n]['potential'][0]
            self.dy[n] = tc.from_numpy(data[n]['gradients'].copy()).double()
            self.pdy[n] = tc.from_numpy(
                          data[n]['pgradients'].copy()).double()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.px[idx], self.y[idx], self.dy[idx],
                self.pdy[idx])


class NeuralNetwork(nn.Module, Model):
    """
    >>>
    >>> import numpy as np
    >>> import torch
    >>> from torch import nn
    >>> from taps.models.torch import NeuralNetwork
    >>> from taps.models import MullerBrown
    >>> from taps.coords import Cartesian
    >>> N = 100
    >>> x = np.linspace(-0.55822365, 0.6234994, N)
    >>> y = np.linspace(1.44172582, 0.02803776, N)
    >>> coords = Cartesian(coords=np.array([x, y]))
    >>> sequential = nn.Sequential(
    ...     nn.Linear(2, 10).double(),
    ...     nn.SiLU().double(),
    ...     nn.Linear(10, 10).double(),
    ...     nn.SiLU().double(),
    ...     nn.Linear(10, 1).double()
    ... )
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> real_model = MullerBrown()
    >>> model = NeuralNetwork(sequential=sequential,
    ... real_model=real_model).to(device)
    >>> model.get_potential(coords=coords).shape
    >>> # model(coords).shape
    ... torch.Size([100, 1])

    """
    implemented_properties = {'potential', 'gradients', 'hessian',
                              'covariance'}

    def __init__(self, sequential=None, **kwargs):
        super(NeuralNetwork, self).__init__()
        self.sequential = sequential
        Model.__init__(self, **kwargs)

    def __call__(self, coords):
        if isinstance(coords, Coordinate):
            coords = tc.from_numpy(coords.coords.T).double()
            return nn.Module.__call__(self, coords).detach().numpy().flatten()
        elif isinstance(coords, np.ndarray):
            coords = tc.from_numpy(coords.T).double()
            return nn.Module.__call__(self, coords).detach().numpy().flatten()

        return nn.Module.__call__(self, coords)

    def __getattr__(self, key):
        if key == 'real_model':
            return self
        else:
            return super().__getattr__(key)

    def forward(self, tensor):
        res = self.sequential(tensor)
        return res

    def calculate(self, coords, properties=None, **kwargs):
        # self.eval()
        D, N = coords.D, coords.N
        tensor = tc.from_numpy(
            coords.coords.reshape(D, N).T).double()
        V = self.forward(tensor)
        if 'potential' in properties:
            self.results['potential'] = V.detach().numpy()[..., 0]
        if 'gradients' in properties:
            dV = tc.zeros((N, D))
            for n in range(N):
                dV[n] = jacobian(self.forward, tensor[n])
            self.results['gradients'] = dV.detach().numpy().T
        if 'hessian' in properties:
            H = tc.zeros((N, D, D))
            for n in range(N):
                H[n] = hessian(self.forward, tensor[n])
            self.results['hessian'] = H.detach().numpy().T
        if 'covariance' in properties:
            self.results['covariance'] = tc.zeros(N).detach().numpy()

    def save_hyperparameters(self, filename):
        tc.save(self.state_dict(), filename)

    def load_hyperparameters(self, filename):
        self.load_state_dict(tc.load(filename))


class AtomicNeuralNetwork(nn.Module, Model):
    """

    atomic_numbers = coords.species.numbers
    sequendict = {}
    moduledict = nn.ModuleDict()
    for an in atomic_numbers:
        if sequendict.get(an) is None:
            sequendict[an] = nn.Sequential(
                nn.Linear(39, 50).double(),
                nn.SiLU().double(),
                nn.Linear(50, 50).double(),
                nn.SiLU().double(),
                nn.Linear(50, 1).double()
            )
        moduledict.append(sequendict[an])

    model = PerAtomicNeuralNetwork(moduledict=moduledict, prj=prj, desc=desc)
    #moduledict[1](tc.from_numpy(dcrds.coords[:, a, np.newaxis]).double())[..., 0, 0].shape
    print(model.get_potential(coords=coords).shape,
          model.get_gradients(coords=coords).shape,
          model.get_hessian(coords=coords).shape)

    """
    implemented_properties = {'potential', 'gradients', 'hessian',
                              'covariance'}

    def __init__(self, moduledict=None, desc=None, numbers=None, timestamps={},
                 **kwargs):
        super(AtomicNeuralNetwork, self).__init__()
        self.moduledict = moduledict
        self.desc = desc
        self.numbers = numbers
        self.timestamps = timestamps
        Model.__init__(self, **kwargs)

    def __call__(self, coords):
        if isinstance(coords, Coordinate):
            coords = tc.from_numpy(coords.coords).double()
            return nn.Module.__call__(self, coords).detach().numpy()
        elif isinstance(coords, np.ndarray):
            coords = tc.from_numpy(coords).double()
            return nn.Module.__call__(self, coords).detach().numpy()

        return nn.Module.__call__(self, coords).sum()

    def __getattr__(self, key):
        if key == 'real_model':
            return self
        else:
            return super().__getattr__(key)

    def forward(self, tensor, cells):
        # Calculation of descriptor
        a = time.time()
        desc = self.desc(tensor, cells)
        b = time.time()
        self.timestamps['descriptor'] = b - a

        # temp = tc.zeros(N, self.A)
        a = time.time()
        temp = []
        for n, spe in enumerate(self.numbers):
            temp.append(self.moduledict[str(spe)](desc[:, n]))
        b = time.time()
        res = tc.cat(temp, axis=1)
        self.timestamps['moduledict'] = b - a
        return res
        # return tc.sum(res, axis=1)

    def V(self, x):
        return tc.sum(self.forward(x), axis=1)

    def dV(self, tensor, V):
        # dV = grad(tc.sum(V), tensor, create_graph=True)[0]
        dV = grad(tc.sum(V), tensor, create_graph=True, allow_unused=True)[0]
        return dV

    def calculate(self, coords, properties=['potential'], vectorize=True,
                  **kwargs):
        """
        coords: N x A x G
        """
        # self.eval()
        N, A, D = coords.N, coords.A, coords.D
        tensor = coords.coords

        results = {}
        potentials = self.forward(tensor)
        # print(potentials)

        a = time.time()
        potential = tc.sum(potentials, axis=1)
        b = time.time()
        self.timestamps['potential'] = b - a

        if 'gradients' in properties or 'hessian' in properties:
            a = time.time()
            gradients = grad(tc.sum(potential), tensor, create_graph=True,
                             allow_unused=True)[0]
            b = time.time()
            self.timestamps['gradients'] = b - a
            a = time.time()
            results['gradients'] = gradients.cpu().detach().numpy()
            b = time.time()
            self.timestamps['gradients.numpy()'] = b - a

        # if 'hessian' in properties:
        #     H = tc.zeros((N, A, 3, A, 3), dtype=tensor.dtype,
        #                  device=tensor.device)
        #     for n in range(N):
        #         H[n] = hessian(self.V, tensor[None, n],
        #                        vectorize=vectorize)[:, :, 0, :, :]
        #     results['hessian'] = H.detach().numpy().reshape(N, D, D)

        if 'hessian' in properties:
            # x = tensor.clone().requires_grad_()
            a = time.time()
            H = tc.zeros((N, A, 3, A, 3), dtype=tensor.dtype,
                          device=tensor.device)
            for n in range(N):
                x = tensor[n].view(-1)
                z = gradients[n].view(-1)
                j = []
                hv, = torch.autograd.grad(g, x, grad_outputs=v, allow_unused=True)
                for i in range(z.nelement()):
                    x.grad = None
                    v = tc.zeros_like(x)
                    v[i] = 1.
                    z.backward(v, retain_graph=True)
                    j.append(x.grad)
                H[n] = tc.stack(j)
            b = time.time()
            self.timestamps['hessian'] = b - a

            a = time.time()
            results['hessian'] = H.cpu().detach().numpy().reshape(N, D, D)
            b = time.time()
            self.timestamps['hessian'] = b - a

        if 'potentials' in properties:
            results['potentials'] = potentials.cpu().detach().numpy()

        if 'potential' in properties:
            results['potential'] = potential.cpu().detach().numpy()

        if 'covariance' in properties:
            self.results['covariance'] = tc.zeros(N).detach().numpy()

        self.results = results

    def save_hyperparameters(self, filename):
        tc.save(self.state_dict(), filename)

    def load_hyperparameters(self, filename):
        self.load_state_dict(tc.load(filename))

    def get_finite_hessian(self, coords=None, paths=None, eps=1e-2, **kwargs):
        """
        coords : NxAx3
        newcoords:
        """
        if coords is None:
            coords = paths.coords

        pcoords = self.prj.x(coords)

        def func(x):
            x0 = pcoords.similar(coords=x)
            self.calculate(coords=x0, properties=['potential'], **kwargs)
            return tc.from_numpy(self.results['potential'].copy())
        with tc.no_grad():
            hess = get_finite_hessian(func, pcoords.coords, eps=eps)
        return hess

    def get_finite_gradients(self, coords=None, paths=None, eps=1e-2,
                             **kwargs):
        """
        coords : NxAx3
        newcoords:
        """
        if coords is None:
            coords = paths.coords

        pcoords = self.prj.x(coords)

        def func(x):
            x0 = pcoords.similar(coords=x)
            self.calculate(coords=x0, properties=['potential'], **kwargs)
            return tc.from_numpy(self.results['potential'].copy())
        with tc.no_grad():
            grad = get_finite_gradients(func, pcoords.coords, eps=eps)
        return grad
