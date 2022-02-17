import sys
import torch
import torch as tc
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
from torch.autograd.functional import hessian
from taps.ml.regressions import Regression
from taps.projectors import Projector
from collections import Counter


class TensorProjector:
    """
    Accept Numpy and return Tensor
    """
    def __init__(self, device=None, **kwargs):
        self.device = device or "cuda" if tc.cuda.is_available() else "cpu"
        super().__init__(**kwargs)

    def x(self, coords):
        tensor = tc.from_numpy(coords.coords.T).to(device=self.device)
        tensor.requires_grad = True
        return tensor

    def _x(self, coords):
        tensor = tc.from_numpy(coords.T).to(device=self.device)
        tensor.requires_grad = True
        return tensor

    def x_inv(self, tensor):
        return tensor.detach().numpy().T

    def _x_inv(self, tensor):
        return tensor.detach().numpy().T

    def V(self, potential):
        tensorV = tc.from_numpy(potential).to(device=self.device)
        tensorV.requires_grad = True
        return tensorV

    def V_inv(self, potential):
        return potential.flatten().detach().numpy()


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

    def __init__(self, database):
        data = database.read_all(entries={'coord': 'blob',
                                          'potential': 'blob'})
        N = len(data)
        D = len(data[0]['coord'].flatten())
        self.x = tc.empty(N, D).double()
        self.y = tc.empty(N).double()
        for n in range(N):
            self.x[n] = tc.from_numpy(data[n]['coord'].copy()).double()
            self.y[n] = data[n]['potential'][0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PyTorchRegression(Regression):
    """
    >>> sequence = nn.Sequential(
    >>>     nn.Linear(2, 512),
    >>>     nn.ReLU(),
    >>>     nn.Linear(512, 512),
    >>>     nn.ReLU(),
    >>>     nn.Linear(512, 1)
    >>> )
    >>> kernel = PyTorchKernel(sequence=sequence).to(device='cpu')
    >>> optimizer =  torch.optim.Adam(kernel.parameters(), lr=1e-3)
    >>> regression = PyTorchRegression(optimizer=optimizer)
    """
    def __init__(self, loss=None, optimizer=None, epochs=None,
                 batch_size=None, device=None, log=None, **kwargs):
        self.loss = loss or nn.MSELoss()
        self.optimizer = optimizer

        self.epochs = epochs or 5
        self.batch_size = batch_size or 1
        self.device = device or "cuda" if tc.cuda.is_available() else "cpu"
        self.log = log or sys.stdout
        super().__init__(**kwargs)

    def train(self, kernel, database, loss=None, optimizer=None, epochs=None,
              batch_size=None, device=None, log=None):
        """
        database: Database class
        """

        loss = loss or self.loss
        optimizer = optimizer or self.optimizer
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        device = device or self.device
        log = log or self.log

        train_dataloader = self.dataloader(database, batch_size=batch_size)

        for t in range(epochs):
            self._train(train_dataloader, kernel, loss, optimizer, device, log)

    def dataloader(self, database, batch_size=None):
        """
        database: Database class
        batch_size: Int

        >>> dataloader = data_loader(database)
        >>> assert isinstance(dataloader)
        """
        batch_size = batch_size or self.batch_size
        dataset = ImageDatabaseDataset(database)
        return DataLoader(dataset, batch_size=batch_size)

    def _train(self, dataloader, kernel, loss, optimizer, device, log):
        size = len(dataloader.dataset)
        kernel.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.view(-1, 1).to(device)

            # Compute prediction error
            pred = kernel(X)
            l = loss(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if batch % 100 == 0:
                l, current = l.item(), batch * len(X)
                log.write(f"loss: {l:>7f}  [{current:>5d}/{size:>5d}] \n")

    def test(dataloader, model, loss_fn):
        # size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: {test_loss:>8f} \n")


class PyTorchKernel(nn.Module):
    """
    >>>
    sequence = nn.Sequential(
    nn.Linear(2, 10).double(),
    nn.SiLU().double(),
    nn.Linear(10, 10).double(),
    nn.SiLU().double(),
    nn.Linear(10, 1).double()
)
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> sequence = nn.Sequential(
    >>>     nn.Linear(inshape, 512),
    >>>     nn.ReLU(),
    >>>     nn.Linear(512, 512),
    >>>     nn.ReLU(),
    >>>     nn.Linear(512, outshape)
    >>> )
    >>> kernel = PyTorchKernel(sequence=sequence, device=device)
    """
    def __init__(self, sequence=None, prj=None):
        super(PyTorchKernel, self).__init__()
        self.sequence = sequence
        self.prj = prj or TensorProjector()

    def forward(self, tensor):
        # tensor = self.flatten(tensor)
        logits = self.sequence(tensor)
        return logits

    def get_potential(self, coords):
        self.eval()
        tensor = self.prj.x(coords)
        V = self.forward(tensor)
        return self.prj.V_inv(V)

    def get_gradients(self, coords):
        self.eval()
        tensor = self.prj.x(coords)
        V = self.forward(tensor)
        return grad(tc.sum(V), tensor, create_graph=True)[0]

    def get_hessian(self, coords):
        D, N = coords.D, coords.N
        self.eval()
        tensor = self.prj.x(coords)
        H = np.zeros((D, D, N))
        for n in range(N):
            H[..., n] = hessian(self.forward, tensor[n]).detach().numpy()

        return H

    def save_hyperparameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_hyperparameters(self, filename):
        self.load_state_dict(torch.load(filename))


class AtomCenteredPyTorchKernel(nn.Module):
    """
    in_channel: Dimension of the descriptor
    out_channel: number of nodes in a hidden layer.
    groups: Number of species
    'H2O' 예제의 경우 -> H O
    if ns is even:
        # a = nn.Conv1d(in_channels=4, out_channels=6, kernel_size=1, groups=ns, stride=2)
    else:
        self.layers = nn.Sequential(
        nn.Conv1d(in_channels=2, out_channels=6, kernel_size=1, groups=2),
        nn.Conv1d(in_channels=6, out_channels=3, kernel_size=1, groups=3))
    self.shared_weight = nn.Parameters(tc.rand(10, 10))
            index = [1, 3, 5, 7, 9]
    #self.fc1.weight = self.shared_weights
    #self.fc2.weight = self.fc2_base_weights.clone()
    #self.fc2.weight[:, index] = self.shared_weights

    """
    def __init__(self, symbols=None, Nd=None, prj=None, **kwargs):
        """
        Nd: Int
            Size of descriptor
        """
        super(AtomCenteredPyTorchKernel, self).__init__(**kwargs)
        self.symbols = symbols
        self.counter = Counter(symbols)
        self.species, self.count = zip(*self.counter.items())

        self.Nd, self.Na, self.Ns = Nd, len(self.symbols), len(self.species)
        self.partition = tc.cumsum(tc.Tensor([0, *self.count]),
                                   dim=0, dtype=int) * self.Nd
        self.prj = prj

        self.layers = []
        for sym, count in self.counter.items():
            self.layers.append(nn.Sequential(
                nn.Conv1d(self.Nd, 120, 1).double(),
                nn.ReLU().double(),
                nn.Conv1d(120, 120, 1).double(),
                nn.ReLU().double(),
                nn.Conv1d(120, 1, 1).double()
            ))

    def forward(self, tensor):
        """
        tensor: Tensor of shape (Batch, in_channel)
        """
        B, C = tensor.shape
        E = 0.
        for i in range(self.Ns):
            init, fin = self.partition[i:i+2]
            E += tc.sum(self.layers[i](tensor[:, init:fin].view(B, self.Nd, -1)))

        return E

    def get_potential(self, coords):
        self.eval()
        tensor = self.prj.x(coords)
        V = self.forward(tensor)
        return self.prj.V_inv(V)

    def get_gradients(self, coords):
        self.eval()
        tensor = self.prj.x(coords)
        V = self.forward(tensor)
        return grad(tc.sum(V), tensor, create_graph=True)[0]

    def get_hessian(self, coords):
        D, N = coords.D, coords.N
        self.eval()
        tensor = self.prj.x(coords)
        H = np.zeros((D, D, N))
        for n in range(N):
            H[..., n] = hessian(self.forward, tensor[n]).detach().numpy()

        return H

    def save_hyperparameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_hyperparameters(self, filename):
        self.load_state_dict(torch.load(filename))
