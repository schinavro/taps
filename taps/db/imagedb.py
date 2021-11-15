from collections import OrderedDict
from taps.db import Database
import sqlite3

from scipy.spatial import KDTree
import time


import numpy as np
from numpy import concatenate as cat
from numpy import newaxis as nax
from taps.utils.antenna import packing
from taps.projectors import Projector

class ImageDatabase(Database):
    """
    Database for image

    TODO: Bug fix on similar data gathering.
    """

    entries=OrderedDict(
        coord='blob',
        label='text',
        status='text',
        start_time='real',
        potential='blob',
        potentials='blob',
        gradients='blob',
        finish_time='real',
        positions='blob',
        forces='blob'
    )
    def __init__(self, data_ids=None, data_bounds=None, table_name='image',
                 prj=None, timeout=60, **kwargs):
        self.data_ids = data_ids or list()
        self.data_bounds = data_bounds or {}
        self.table_name = table_name
        self.prj = prj or Projector()
        self.timeout = timeout
        super().__init__(**kwargs)

    def read(self, ids, prj=None, table_name=None, entries=None, **kwargs):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read(ids,
                            table_name=table_name, entries=entries, **kwargs)
        return data


    def read_all(self, prj=None, table_name=None, entries=None, **kwargs):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read_all(table_name=table_name, entries=entries,
                                **kwargs)
        return data

    def write(self, data, prj=None, table_name=None, entries=None, **kwargs):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name

        return super().write(data, table_name=table_name,
                             entries=entries, **kwargs)

    def update(self, ids, data, prj=None, table_name=None, entries=None,
               **kwargs):

        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name

        super().update(ids, data, table_name=table_name, entries=entries,
                       **kwargs)

    def delete(self, ids, prj=None, table_name=None, entries=None, **kwargs):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name

        super().delete(ids, table_name=table_name, entries=entries, **kwargs)

    def project(self, data, prj=None):
        prj = prj or self.prj
        new_data = None
        for dat in data:
            dat['coords'] = dat['coords']

    def get_image_data(self, paths, prj=None, data_ids=None, data_bounds=None):
        """
        data_ids : dictionary of lists
            {'image': [...], 'descriptor': [...]}
        for each key, return
         'coords' -> 'X'; D x M
         'potential'    -> 'V'; M
         'gradient'    -> 'F'; D x M
        return : dictionary contain X, V, F
            {'X' : np.array(D x M), 'V': np.array(M), }
        """
        prj = prj or self.prj
        data_ids = data_ids or self.data_ids
        data_bounds = data_bounds or getattr(self, 'data_bounds', {})

        shape = prj.x(paths.coords(index=[0])).shape[:-1]
        # Initial state
        if self._cache.get('old_data_ids') is None:
            shape_raw = paths.coords(index=[0]).shape[:-1]
            self._cache['old_data_ids'] = []
            self._cache['data'] = {
                'coords': np.zeros((*shape, 0), dtype=float),
                'potential': np.zeros(0, dtype=float),
                'gradients': np.zeros((*shape, 0), dtype=float),
                'coords_raw': np.zeros((*shape_raw, 0), dtype=float),
                'gradients_raw': np.zeros((*shape_raw, 0), dtype=float),
                'changed': True,
            }

        if self._cache.get('old_data_ids') == data_ids:
            self._cache['data']['changed'] = False
            return self._cache['data']
        else:
            self._cache['data']['changed'] = True
            new_data_ids = []
            for id in data_ids:
                if id not in self._cache['old_data_ids']:
                    new_data_ids.append(id)
        new_data = self.read(new_data_ids)
        M = len(new_data_ids)
        data = self._cache['data']
        keys = list(new_data[0].keys())
        if 'coord' in keys:
            coords_raw = []
            coords_prj = []
            for i in range(M):
                coord_raw = new_data[i]['coord'][..., nax]
                coord_prj = prj._x(new_data[i]['coord'][..., nax])
                coords_raw.append(coord_raw)
                coords_prj.append(coord_prj)
            if M != 0:
                new_coords_raw = cat(coords_raw, axis=-1)
                new_coords_prj = cat(coords_prj, axis=-1)

                data['coords'] = cat([data['coords'],
                                      new_coords_prj], axis=-1)
                data['coords_raw'] = cat([data['coords_raw'],
                                          new_coords_raw], axis=-1)
        if 'potential' in keys:
            potential = []
            for i in range(M):
                new_pot = new_data[i]['potential']
                # Bounds for pot
                if data_bounds.get('potential') is not None:
                    ub = data_bounds['potential'].get('upperbound')
                    lb = data_bounds['potential'].get('lowerbound')
                    if ub is not None and new_pot[0] > ub:
                        print('Potential ub fix')
                        new_pot = [ub]
                    elif lb is not None and new_pot[0] < lb:
                        print('Potential lb fix')
                        new_pot = [lb]
                potential.append(new_pot)
            if M != 0:
                new_potential = np.concatenate(potential, axis=-1)
                data['potential'] = np.concatenate([data['potential'],
                                                    new_potential], axis=-1)
        if 'gradients' in keys:
            gradients_raws = []
            gradients_prjs = []
            for i in range(M):
                coords_raw = new_data[i]['coord'][..., nax]
                gradients_raw = new_data[i]['gradients'].copy()
                gradients_prj, _ = prj.f(gradients_raw, coords_raw)

                if data_bounds.get('gradients') is not None:
                    ub = data_bounds['gradients'].get('upperbound')
                    lb = data_bounds['gradients'].get('lowerbound')
                    if ub is not None:
                        if np.any(gradients_raw > ub):
                            print('Gradients ub fix')
                        gradients_raw[gradients_raw > ub] = ub
                        gradients_prj[gradients_prj > ub] = ub
                    elif lb is not None:
                        if np.any(gradients_raw < lb):
                            print('Gradients lb fix')
                        gradients_raw[gradients_raw < lb] = lb
                        gradients_prj[gradients_prj < lb] = lb
                gradients_raws.append(gradients_raw)
                gradients_prjs.append(gradients_prj)
            if M != 0:
                new_grad_raw = cat(gradients_raws, axis=-1)
                new_grad_prj = cat(gradients_prjs, axis=-1)
                data['gradients_raw'] = cat([data['gradients_raw'],
                                              new_grad_raw], axis=-1)
                data['gradients'] = cat([data['gradients'],
                                         new_grad_prj], axis=-1)
        self._cache['old_data_ids'].extend(new_data_ids)
        data['kernel'] = self.kernel_data(data)
        data['mean'] = self.mean_data(data)
        return data

    def kernel_data(self, data):
        shape = data['coords'].shape
        D, M = np.prod(shape[:-1]), shape[-1]
        X = data['coords'].reshape(D, M)
        Y = cat([data['potential'], data['gradients'].flatten()], axis=0)
        return {'X': X, 'Y': Y}

    def mean_data(self, data):
        return self.kernel_data(data)

    def add_data_ids(self, ids, overlap_handler=True):
        """
        ids : dict of list
        """
        if isinstance(ids, int):
            ids = [ids]
        if overlap_handler:
            for id in ids:
                if id not in self.data_ids:
                    self.data_ids.append(int(id))
        else:
            self.data_ids.extend(ids)

    def add_data(self, paths, coords=None, return_data=False, **kwargs):
        """
        check_overlap -> create datum -> add_data
        coords : atomic configuration
        datum : atomic configuration with potential and forces
           shape,  A number of tuples
             [('H', desc, displacement, potential, forces, directory),
              ('C', desc, ...),
              ('N', desc, ...), ...]
        found : list of Boolean or None, search results
        id : list of int or None, where it is
        M : Number of data
        arr_dict : dict contains pathways to be calculated
              arr_dict['image'] = [np.array([...]), ..., np.array([...])]
              arr_dict['descriptor']
                  = { 'H': [np.array([...]), ..., np.array([...])],
                      'C': [...],
                      ...}
        """

        ids = self.search_image(paths, coords, pack_null=True, **kwargs)
        while True:
            # Not existing data + Failed data
            ids_ntbc = self.get_ids_need_to_be_calculated(ids)
            arr_ntbc = self.get_arr_dict_need_to_be_calculated(ids_ntbc)
            # you_need_calculation = self.check_calculation_is_necessary(ids)

            try:
                self.queue(ids_ntbc, arr_ntbc, **kwargs)
                data = self.create_image_data(paths, arr_ntbc, **kwargs)
                self.update(ids_ntbc, data)

            except Exception as e:
                self.queue(ids_ntbc, arr_ntbc, status='Failed', **kwargs)
                raise NotImplementedError()
                # raise Exception(str(e)) from e
            self.update(ids_ntbc, data, **kwargs)
            if self.You_and_I_have_unfinished_business(ids, **kwargs):
                time.sleep(5)
            else:
                break
        if return_data:
            return self.read(ids)
        return ids

    def search_image(self, paths, coords, pack_null=False,
                     search_similar_image=True, similar_image_tol=0.05,
                     blackids=None, **kwargs):
        """
        Only search image table positions_arr exists.
        coords: 3 x A x M  or  D x M  array
                where M is the number of data and A is the number of image
        If pack null given, fill up the void into slot where no data found.
        Search perfect match exist,
        if not, check similar structure, or lower distance calculate
        """

        db_data = []
        # Initialize
        if search_similar_image:
            entries = OrderedDict(coord='blob', rowid='int')
            db_data = self.read_all(entries=entries)
            if db_data != []:
                db_coord_list = []
                db_coord_flat_list = []
                db_ids = []
                for data in db_data:
                    _coord = data['coord']
                    _id = data['rowid']
                    db_coord_list.append(_coord)
                    db_coord_flat_list.append(_coord.flatten())
                    db_ids.append(_id)
                dbtree = KDTree(db_coord_flat_list)

            paths_ids = self.data_ids
            if len(paths_ids) != 0:
                paths_data = self.get_image_data(paths)
                paths_coord_flat_list = []
                paths_coord_list = paths_data['X'].T
                for m in range(paths_data["X"].shape[-1]):
                    paths_coord_flat_list.append(paths_coord_list[m].flatten())
                pathstree = KDTree(paths_coord_flat_list)

        M = coords.shape[-1]
        ids = []
        for m in range(M):
            conn = sqlite3.connect(self.filename)
            c = conn.cursor()
            coord = coords[..., m]
            # Check Perfect match
            select_statement = "SELECT rowid FROM image WHERE "
            select_statement += "coord=?"
            c.execute(select_statement, [packing(coord)])
            id = c.fetchone()
            conn.commit()
            conn.close()
            if id is None:
                id = []
            elif isinstance(id, tuple):
                id = list(id)

            # Check Similar match
            if search_similar_image and db_data != []:
                # Query similar point exists in db
                res = dbtree.query_ball_point(coord.flatten(),
                                              similar_image_tol)
                similar_ids = np.array(db_ids)[res]
                id.extend(similar_ids)

            # Check any id is in the black list

            if blackids is not None:
                id = [i for i in id if i not in blackids]

            if id != []:
                # pick among fresh ids
                id = [np.random.choice(id)]
            elif id == [] and pack_null:
                # Emergency case! Randomly walk until it finds no overlap
                if len(paths_ids) != 0:
                    prjcoord = paths.model.prj._x(coord[..., nax])
                    shape = prjcoord.shape
                    new_coord = prjcoord.flatten()
                    _D = np.prod(shape)
                    while True:
                        dist, prxi = pathstree.query(new_coord)
                        if dist > similar_image_tol:
                            break
                        walk = np.random.normal(size=_D,
                                                scale=2*similar_image_tol)
                        new_coord += walk
                    _coord = paths.model.prj._x_inv(new_coord.reshape(*shape))
                    coord = _coord[..., 0]

                # PAD empty slots
                data = self.create_vacant_data(coord[..., nax], **kwargs)
                id = self.write(data)
            elif id == []:
                # pick among fresh ids
                id = []

            ids.extend(id)
        return ids

    def create_vacant_data(self, coords, prj=None, entries=None,
                           table_name=None, **kwargs):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name

        properties = entries.keys()
        M = coords.shape[-1]

        image_data = []
        for m in range(M):
            datum = OrderedDict(zip(properties, [None] * len(properties)))
            coord = coords[..., m]
            datum['coord'] = coord
            image_data.append(datum)

        return image_data

    def create_image_data(self, paths, coords, prj=None, entries=None,
                          table_name=None, **kwargs):
        """
        coords : DxM array
        update_status : list of ids
        return : list of tuples containing
              (coord, label, status, start_time, potential, potentials,
               gradients, finish_time, positions, forces)
        """
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name

        properties = entries.keys()
        model_properties = paths.model.real_model.implemented_properties
        props = [p for p in properties if p in model_properties]
        M = coords.shape[-1]

        image_data = []
        for m in range(M):
            datum = OrderedDict(zip(properties, [None] * len(properties)))
            coord = coords[..., m]
            datum['coord'] = coord
            datum['start_time'] = None
            datum['status'] = None

            results = paths.get_properties(coords=coord[..., nax],
                                           properties=props,
                                           real_model=True, caching=True)
            datum['finish_time'] = time.time()

            for key, val in results.items():
                datum[key] = val

            datum['status'] = 'Finished'
            image_data.append(datum)

        return image_data

    def get_arr_dict_need_to_be_calculated(self, ids):
        # For readable code, indexing with string. Not number
        data_image = self.read(ids, entries={'coord': 'blob'})

        coord_T_list = []
        for data in data_image:
            coord_T_list.append(data['coord'].T)
        arr_ntbc = np.array(coord_T_list).T

        return arr_ntbc

    def get_ids_need_to_be_calculated(self, ids):
        """
        Given ids that spans all ids, check if calculation is necessary.
        if it is true
        First, check the status is None
        ids : dictionary of lists that contains id
               {'image': [1, 2, ..], }
        """
        # For readable code, indexing with string. Not number
        ids_ntbc = []
        image_data = self.read(ids)
        # data : list, datum : tuple
        for id, datum in zip(ids, image_data):
            status = datum['status']
            if status is None or status == 'Failed':
                ids_ntbc.append(id)
            elif status == 'Finished':
                continue
            elif status == 'Running':
                # Time out
                if time.time() - datum['start_time'] > self.timeout:
                    ids_ntbc.append(id)
                else:
                    continue
            else:
                raise NotImplementedError('Can not reconize %s' % status)
        return ids_ntbc

    def queue(self, ids, arr_dict, status='Running', **kwargs):
        """
        arr_dict {'image': dim x A x N arr}
        """
        ids_image = ids
        coords = arr_dict
        M = coords.shape[-1]
        for m in range(M):
            datum = OrderedDict(
                coord=coords[..., m], label=None, status=status,
                start_time=time.time(), potential= None, potentials=None,
                gradients=None, finish_time=None, positions=None, forces=None
            )
            id = ids_image[m]
            self.update([id], [datum])

    def You_and_I_have_unfinished_business(self, ids, **kwargs):
        """
        Kill Bill
        Check calculation of given ids are finished
        """
        intrim_report = []
        image_data = self.read(ids)
        statuses = np.array([datum['status'] for datum in image_data])
        if np.all(statuses == 'Finished'):
            intrim_report.append(False)
        else:
            return True
        return np.any(intrim_report)


class AtomicImageDatabase(ImageDatabase):
    table=OrderedDict(
        atoms='blob',
        symbols='text',
        image_number='integer',
        cell='blob',
        coord='blob',
        label='text',
        status='text',
        start_time='real',
        potential='blob',
        potentials='blob',
        gradients='blob',
        finish_time='real',
        positions='blob',
        forces='blob'
    )
