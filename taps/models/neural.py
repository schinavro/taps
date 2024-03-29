from taps.models import Model


class NeuralNetwork(Model):
    """ Simple neural network potentials
    Neural network potential.
    %pylab inline
    import numpy as np
    import torch
    import torch as tc
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from taps.coords import Cartesian
    from taps.paths import Paths
    from taps.models import MullerBrown

    #from torchvision import datasets
    #from torchvision.transforms import ToTensor
    xlim = tc.Tensor([-1.5, 1])
    ylim = tc.Tensor([-0.5, 2])
    ip = tc.Tensor([xlim[0], ylim[0]])
    fp = tc.Tensor([xlim[1], ylim[1]])

    x = (tc.rand(100000, 2, dtype=tc.float32))*(fp - ip) + ip
    test_x = (tc.rand(100, 2, dtype=tc.float32))*(fp - ip) + ip

    model = MullerBrown()
    model.calculate(None, Cartesian(coords=x.numpy().T))

    y = tc.from_numpy(model.results['potential'])

    model.calculate(None, Cartesian(coords=test_x.numpy().T))
    test_y = tc.from_numpy(model.results['potential'])
    from taps.models import MullerBrown
    from taps.ml.torch import PyTorchKernel, PyTorchRegression
    from taps.models.neural import NeuralNetwork

    from torch import nn

    import torch
    sequence = nn.Sequential(
        nn.Linear(2, 512).double(),
        nn.ReLU().double(),
        nn.Linear(512, 512).double(),
        nn.ReLU().double(),
        nn.Linear(512, 1).double()
    )

    real_model = MullerBrown()
    kernel = PyTorchKernel(sequence=sequence).to(device='cpu')

    optimizer =  torch.optim.Adam(kernel.parameters(), lr=1e-3)
    regression = PyTorchRegression(optimizer=optimizer, batch_size=10000)
    model = NeuralNetwork(#real_model=real_model,
                          kernel=kernel,
                          regression=regression)
    import numpy as np
    from taps.coords import Cartesian
    coords = Cartesian(coords=np.ones((2, 50)))
    model.get_potential_energy(None, coords=coords)
    from taps.db import ImageDatabase
    database = ImageDatabase(filename='test2.db')
    #list_dat = [dict(coord=coord, potential=[pot])
    #            for coord, pot in zip(x.numpy(), y.numpy()[:])]
    #database.write(data=list_dat)
    model.regression(model.kernel, database)
    from taps.paths import Paths
    from taps.visualize import view
    paths = Paths(coords=coords, model=model)
    view(paths, viewer='MullerBrown')

    >>> from
    >>> from taps.ml.torch import PyTorchKernel, PyTorchRegression
    >>> from taps.models.neural import NeuralNetwork
    >>> model = NeuralNetwork(real_model=)

    def save_hyperparameters(self, filename):
        self.kernel.save_hyperparameters(filename)

    def load_hyperparameters(self, filename):
        self.kernel.load_hyperparameters(filename)


    """
    implemented_properties = ['potential', 'gradients', 'hessian']

    def __init__(self, kernel=None, **kwargs):
        self.kernel = kernel

        super().__init__(**kwargs)

    def __call__(self, coords):
        return self.get_potential_energy(None, coords=coords)

    def calculate(self, paths, coords, properties=None, **kwargs):
        if 'potential' in properties:
            self.results['potential'] = self.kernel.get_potential(coords)
        if 'gradients' in properties:
            self.results['gradients'] = self.kernel.get_gradients(coords)
        if 'hessian' in properties:
            self.results['hessian'] = self.kernel.get_hessian(coords)


def train(model, loss_fn, x0, database, optimizer, epochs=5,
          batch_size=10, device='cpu', log=sys.stdout):
    """
    PyTorch NN Regression
    """
    train_dataloader = dataloader(database, batch_size=batch_size)

    for t in range(epochs):
        # _train(train_dataloader, kernel, loss, optimizer, device, log)
        size = len(train_dataloader.dataset)
        kernel.train()
        for batch, (X, y) in enumerate(train_dataloader):
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


def dataloader(database, batchsize):
    """
    database: Database class
    batch_size: Int

    >>> dataloader = data_loader(database)
    >>> assert isinstance(dataloader)
    """
    dataset = ImageDatabaseDataset(database)
    return DataLoader(dataset, batch_size=batch_size)

    
