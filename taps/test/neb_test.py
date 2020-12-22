import numpy as np

from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton

from taps.paths import Paths
from taps.model.atomicmodel import AtomicModel
from taps.projector import Mask
from taps.pathfinder import NEB

slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]

_slab = slab.copy()
slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.calc = EMT()

# Initial state:
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# init = slab.positions[mask].T
init = slab.positions.T.copy()

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)

fin = qn.atoms.positions.T
slab.constraints = []

model = AtomicModel(image=slab)
prj = Mask(mask=~np.array(mask), orig_coord=_slab.positions.T)
finder = NEB(prj=prj, iter=10)

paths = Paths(coords=coords)
paths.model = model
paths.finder = finder

N = 150
dist = fin - init

coords = np.linspace(0, 1, N)[np.newaxis, :] * dist[:, :, np.newaxis]  \
         + init[:, :, np.newaxis]

paths.search()
