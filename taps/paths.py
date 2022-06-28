import copy
import pickle
import numpy as np



class Paths:
    """ Paths class infterfacing other modules

    Centeral class for pathway calculation.
    Object contains coordinates class and potential class and database

    Parameters
    ----------

    coords  : Cartesian class
        Coordinate representation that connect initial state to final state.
        Calculation involving intermediate states, such as, kinetic energy
        or momentum, can be found in the Cartesian class.
    label   : string
        Set of character that distinguish from other Paths classes. This can be
        use in other module as default filename.
    model   : Model class
        Model that calculates potential energy of intermediate coordinates.
    finder  : PathFinder class
        Class that optimize the coordinates suitable for the finder class.
    imgdb : Database class
        Class made to easily accessible to the calculated database.
    tag     : dict
        Auxilary parameters for external use.

    Example
    -------

    >>> import numpy as np
    >>> from taps.paths import Paths
    >>> x = np.linspace(-0.55822365, 0.6234994, 300)
    >>> y = np.linspace(1.44172582, 0.02803776, 300)
    >>> paths = Paths(coords = np.array([x, y]))

    >>> from taps.models.mullerbrown import MullerBrown
    >>> paths.model = MullerBrown()

    >>> from taps.visualize import view
    >>> view(paths, calculate_map=True, viewer='MullerBrown')
    [Map should be shown]
    """

    def __init__(self, coords=None, label=None, model=None,
                 finder=None, imgdb=None, tag=None, **kwargs):

        self.coords = coords
        self.label = label

        self.model = model
        self.finder = finder
        self.imgdb = imgdb
        self.tag = tag

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_kinetics(self, **kwargs):
        return self.coords.get_kinetics(paths=self, **kwargs)

    def get_masses(self, **kwargs):
        return self.coords.masses(paths=self, **kwargs)

    def get_distances(self, **kwargs):
        """ Calculate the distances of the pathway (accumulative)
        It usually used for plotting, i.e. potential energy / distance
        Calls the :meth:`get_distances` at the ``paths.model``.
        Model calls the :meth:`get_distances` in the ``paths.coords``.
        """
        return self.coords.get_distances(paths=self, **kwargs)

    def get_speeds(self, **kwargs):
        return self.coords.get_speeds(paths=self, **kwargs)

    def get_displacements(self, **kwargs):
        return self.coords.get_displacements(paths=self, **kwargs)

    def get_velocities(self, **kwargs):
        r""" Calculate velocity
        If :class:`Cartesian` is cartesian, velocity is calculated via
        :math:`\mathbf{x}_{i+1} - \mathbf{x}_{i}`"""
        return self.coords.get_velocities(paths=self, **kwargs)

    def get_accelerations(self, **kwargs):
        """ Calculate acceleration
        If :class:`Cartesian` is cartesian, velocity is calculated via
        """
        return self.coords.get_accelerations(paths=self, **kwargs)

    def get_momentums(self, **kwargs):
        """ Calculate momentum of the pathway"""
        return self.coords.get_momentums(self, **kwargs)

    def get_kinetic_energies(self, **kwargs):
        """ Calculate kinetic energy of the pathway"""
        return self.coords.get_kinetic_energies(paths=self, **kwargs)

    def get_kinetic_energy_gradients(self, **kwargs):
        r""" Calculate kinetic energy gradient.
        differentiate Kinetic energy w.r.t. each point, That is we calculate
        :math: `\partial_{\mathbf{x}}E_{\mathrm{kin}}`
        """
        return self.coords.get_kinetic_energy_gradients(paths=self, **kwargs)

    def get_properties(self, **kwargs):
        """ Directly calls the :meth:`get_properties` in ``paths.model``"""
        return self.model.get_properties(paths=self, **kwargs)

    def get_potential_energy(self, **kwargs):
        """ Calculate potential( energy) """
        return self.model.get_potential_energy(paths=self, **kwargs)

    def get_potential(self, **kwargs):
        """ Calculate potential """
        return self.model.get_potential(paths=self, **kwargs)

    def get_potential_energies(self, **kwargs):
        """ Equivalanet to Calculate potentials"""
        return self.model.get_potential_energies(paths=self, **kwargs)

    def get_potentials(self, **kwargs):
        """ Calculate potentials, individual energy of each atoms"""
        return self.model.get_potentials(paths=self, **kwargs)

    def get_forces(self, **kwargs):
        """ Calculate - potential gradient"""
        return self.model.get_forces(paths=self, **kwargs)

    def get_gradient(self, **kwargs):
        """ Calculate potential gradient"""
        return self.model.get_gradient(paths=self, **kwargs)

    def get_gradients(self, **kwargs):
        """ Calculate potential gradient(s)"""
        return self.model.get_gradients(paths=self, **kwargs)

    def get_hessian(self, **kwargs):
        """ Calculate Hessian of a potential"""
        return self.model.get_hessian(paths=self, **kwargs)

    def get_total_energy(self, **kwargs):
        """ Calculate kinetic + potential energy"""
        V = self.model.get_potential_energy(paths=self, **kwargs)
        T = self.model.get_kinetic_energies(paths=self, **kwargs)
        return V + T

    def get_covariance(self, **kwargs):
        """ Calculate covariance. It only applies when potential is guassian"""
        return self.model.get_covariance(paths=self, **kwargs)

    def get_higest_energy_idx(self):
        """ Get index of highest potential energy simplified"""
        E = self.get_potential_energy()
        return np.argmax(E)

    def get_lowest_confident_idx(self):
        """ Get index of lowest of covariance simplified"""
        cov = self.model.get_covariance(self)
        return np.argmax(np.diag(cov))

    def get_image_data(self, **kwargs):
        """" list of int; Get rowid of data"""
        return self.imgdb.get_image_data(paths=self, **kwargs)

    def add_image_data(self, index=None, coords=None, force=False,
                       cache_model=True, **kwargs):
        """ Adding a calculation data to image database

        if index given -> create coords -> add_image_data
        if coords given -> add_image_data
        coords : atomic configuration
        datum : atomic configuration with energy and forces
           shape,  A number of tuples
             [('H', desc, displacement, potential, forces, directory),
              ('C', desc, ...),
              ('N', desc, ...), ...]
        found : list of Boolean or None, search results
        id : list of int or None, where it is
        """
        if coords is None:
            # Create positional descriptor
            coords = self.coords(index=index)
        ids = self.imgdb.add_image_data(self, coords, force=force)
        if cache_model:
            self.imgdb.add_image_data_ids(ids)
        return ids

    def fluctuate(self, *args, **kwargs):
        self.coords.fluctuate(*args, **kwargs)

    def search(self, **kwargs):
        """ Calculate optimized pathway"""
        self.finder.search(self, **kwargs)

    def to_pickle(self, filename=None, simple=True):
        """ Saving pickle simplified"""
        if simple:
            filename = filename or self.label + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(f, self)

    def copy(self):
        """Return a copy."""
        return copy.deepcopy(self)

    def save(self, *args, **kwargs):
        """
        Different method of writing is required for different models.
        """
        self.model.save(self, *args, **kwargs)
