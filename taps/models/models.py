import copy
import hashlib
import binascii
import numpy as np
from pathlib import Path
from taps.projectors import Projector


class Model:
    """ Model for static property calculation

    Parameters
    ----------

    real_model: Model class
       Model for gaining data for machine learning purpose.
    results: dict
       dictionary saves calculated results.
    label: str
       name of calculation
    directory: str
       directory of the label
    prefix: str
       name of the label
    potential_unit: str
       default; eV
    data_ids: dict
       dictionary contains id of the data.
       In order to save the data, use
       >>> paths.add_image_data(index=[0, -1]) # create&save init and fin image
       >>> paths.imgdb.data_ids["image"]
       [1, 2, 15, ...]
    optimized: bool
       Bool check hyperparameters
    prj: Projector class

    """
    implemented_properties = {'potential'}
    unit = 'eV'

    def __init__(self, results=None, label=None, prj=None, _cache=None,
                 unit='eV'):
        self.results = results or {}
        self.label = label
        self.prj = prj or Projector()
        self.unit = unit
        self._cache = _cache or {}

    def __getattr__(self, key):
        if key == 'real_model':
            return self
        else:
            super().__getattribute__(key)

    def __call__(self, coords):
        return self.get_potential(coords=coords)

    def get_properties(self, paths=None, properties=['potential'],
                       index=np.s_[:], coords=None, caching=False,
                       real_model=False, return_dict=False, **kwargs):
        """ pre-calculation rutine.

        Calculate static related properties.

        Parameters
        ----------

        properties: list of str
            'potential', 'potentials', 'gradients' and 'hessian'.
        index: slice obj
            default; np.s_[1:-1]
        coords: Cartesian class
        caching: bool; default false
        real_model: bool; default False
           choose use real_model or not
        """
        if type(properties) == str:
            properties = [properties]
        # For machine learning purpose we put real_model in model
        if real_model:
            model = self.real_model
        else:
            model = self

        for property in properties:
            if property not in model.implemented_properties:
                raise NotImplementedError('Can not calaculate %s' % property)
        if coords is None:
            coords = paths.coords(index=index)

        new_coords = None
        new_properties = []
        results = {}
        if caching:
            model._cache = getattr(model, '_cache', dict())
            cached_results = model._cache.get(coords.tobytes())
            if cached_results is not None:
                cached_properties = cached_results.keys()
                for property in properties:
                    if property in cached_properties:
                        results[property] = cached_results[property]
                    else:
                        new_properties.append(property)
            else:
                new_properties = list(properties)
                new_coords = coords
        else:
            new_properties = properties
            new_coords = coords

        if new_coords is not None:
            model.calculate(new_coords, paths=paths, properties=new_properties,
                            **kwargs)

        for new_property in new_properties:
            results[new_property] = model.results[new_property]

            if new_property == 'gradients':
                res = results[new_property]
                if caching:
                    model.results['pgradients'] = res.copy()
                results[new_property] = model.prj.f_inv(res, new_coords)[0]
            elif new_property == 'potential':
                res = results[new_property]
                if caching:
                    model.results['ppotential'] = res.copy()
                results[new_property] = model.prj.V_inv(res)

            elif new_property == 'hessian':
                res = results[new_property]
                if caching:
                    model.results['phessian'] = res.copy()
                results[new_property] = model.prj.h_inv(res, new_coords)[0]

        if caching:
            model._cache[coords.tobytes()] = copy.deepcopy(results)
        if len(properties) == 1 and not return_dict:
            property = list(results.keys())[0]
            return results[property]
        return results

    def get_potential(self, **kwargs):
        return self.get_properties(properties='potential', **kwargs)

    def get_potentials(self, **kwargs):
        return self.get_properties(properties='potentials', **kwargs)

    def get_potential_energy(self, **kwargs):
        return self.get_properties(properties='potential', **kwargs)

    def get_potential_energies(self, **kwargs):
        return self.get_properties(properties='potentials', **kwargs)

    def get_forces(self, **kwargs):
        return self.get_properties(properties='forces', **kwargs)

    def get_gradient(self, **kwargs):
        return self.get_properties(properties='gradients', **kwargs)

    def get_gradients(self, **kwargs):
        return self.get_properties(properties='gradients', **kwargs)

    def get_hessian(self, **kwargs):
        return self.get_properties(properties='hessian', **kwargs)

    def get_covariance(self, **kwargs):
        return self.get_properties(properties='covariance', **kwargs)

    def generate_unique_hash(self, positions):
        """return string that explains current calculation
        """
        dk = hashlib.pbkdf2_hmac('sha512', b"Heyy!C@reful!ThisisMyP@ssword!",
                                 positions.tobytes(), 1234)
        return str(binascii.hexlify(dk))[2:-1]

    def get_label(self, coord=None):
        path = Path(self.label)
        directory, name = str(path.parent), path.name
        unique_hash = self.generate_unique_hash(coord)
        return directory + '/' + unique_hash + '/' + name

    def get_directory(self, coord=None):
        path = Path(self.label)
        directory = str(path.parent)
        unique_hash = self.generate_unique_hash(coord)
        return directory + '/' + unique_hash + '/'

    def get_state_info(self):
        return ""

    def save(self, paths, *args, real_model=None, coords=None, index=np.s_[:],
             **kwargs):
        if real_model:
            model = self.real_model
        else:
            model = self

        if coords is None:
            coords = model.prj.x(paths.coords(index=index))

        model.write(paths, coords, **kwargs)
