import os
import time
import random
import copy
import pickle
import sqlite3
from sqlite3 import OperationalError
import numpy as np
from ase.utils import Lock
from collections import OrderedDict
from taps.utils import dct2pd_dct, dfseries2dct
from taps.utils import isStr, isbool

import ase.io.jsonio
from ase.db.core import (connect, bytes_to_object, object_to_bytes)

from taps.descriptor import SphericalHarmonicDescriptor


def concatenate_atoms_data(filename, d1, d2, delunary_triangle=False):
    """
    concatenate two atoms_data and make new one.
    """

    def _random(idx, partition):
        idx = np.random.choice(idx, len(idx) // 2, replace=False)
        idx[idx < partition], idx[idx >= partition]
        return idx[idx < partition], idx[idx >= partition] - partition

    def _uniform(x, idx, partition):
        raise NotImplementedError('Not')

    d3 = connect(filename)
    id_d1, x_d1 = [], []
    id_d2, x_d2 = [], []
    for row1 in d1.select('id!=-1', columns=['id'], include_data=True):
        id_d1.append(row1.id)
        x_d1.append(row1.data.X)
    partition = len(id_d1)
    for row2 in d2.select('id!=-1', columns=['id'], include_data=True):
        id_d2.append(row2.id)
        x_d2.append(row2.data.X)

    try:
        d1_calculator = True
        # attach_calculator=True, add_additional_information=True
        row1.toatoms(d1_calculator, True)
    except AttributeError:
        d1_calculator = False
    try:
        d2_calculator = True
        row2.toatoms(d2_calculator, True)
    except AttributeError:
        d2_calculator = False

    x_12 = np.array(x_d1 + x_d2)
    x, idx = np.unique(np.array(x_12), return_index=True, axis=0)
    if delunary_triangle:
        idx_1, idx_2 = _uniform(x, idx, partition)
    else:
        idx_1, idx_2 = _random(idx, partition)

    id_d1_heritage = np.array(id_d1)[idx_1]
    id_d2_heritage = np.array(id_d2)[idx_2]

    with d3 as db:
        for id1 in id_d1_heritage:
            for row in d1.select('id=%d' % id1, limit=2):
                atoms = row.toatoms(d1_calculator, True)
                data = row.data
                kvp = row.key_value_pairs
                db.write(atoms, key_value_pairs=kvp, data=data)
        for id2 in id_d2_heritage:
            for row in d2.select('id=%d' % id2, limit=2):
                atoms = row.toatoms(d2_calculator, True)
                data = row.data
                kvp = row.key_value_pairs
                db.write(atoms, key_value_pairs=kvp, data=data)
    d3.metadata = d1.metadata
    return d3


class DataOverlapError(Exception):
    pass


class DataNotExistsError(Exception):
    pass


class PathsData:
    import pandas as pd

    def __init__(self, filename):
        self.filename = filename
        self.df = self.read(self.filename)

    @property
    def filename(self):
        return self.label + '.' + self.format

    @filename.setter
    def filename(self, filename):
        try:
            label, format = filename.rsplit('.', 1)
        except ValueError:
            label = filename
            format = 'csv'
        formats = ['csv', 'pkl']
        if format not in formats:
            label = filename
            format = 'csv'
        self.label, self.format = ''.join(label), format

    def write(self, candidates, mode='w', index=True, header=True,
              **additional_kwargs):
        """paths_dct = paths.__dict__
        df = pd.DataFrame({'coords': [A.flatten()]})
        coords = df.loc[[0]]['coords'].values[0]
        dct = {}
        for key in property:
            dcf[key] = paths_dct[key]"""
        if candidates.__class__.__name__ == 'Paths':
            candidates = [candidates]
            mode == 'a'
        ids, dfs = [], []
        count = len(self.df.index)
        for i, paths in enumerate(candidates):
            if mode == 'a':
                i += count
            s, c, dct = paths.copy(return_dict=True)
            pd_dct = dct2pd_dct(dct)
            pd_dct.update({'id': i})
            pd_dct.update(additional_kwargs)
            ids.append(i)
            dfs.append(self.pd.DataFrame({'symbols': str(s),
                                         'coords': [c.tolist()], **pd_dct},
                                         index=[i]))
        if mode == 'w':
            self.df = self.pd.concat(dfs)
            self.save(df=self.df, mode='w', index=True, header='True')
        elif mode == 'a':
            dataframe = self.pd.concat(dfs)
            # self.df = self.pd.concat([self.df, dataframe], ignore_index=True)
            # self.save(df=dataframe, index=index, mode='a', header='False',
            #           **additional_kwargs)
            dataframe.to_csv(self.filename, mode='a', index=True, header=False)
        return ids

    def update(self, candidates, update_variables=['coords']):
        """
        Update method does not support partial writting. If data is too big,
        frequent update may decrease the performance.

        """
        if candidates.__class__.__name__ == 'Paths':
            candidates = [candidates]
        for variable in update_variables:
            for paths in candidates:
                id = paths.id
                self.df.at[id, variable] = [getattr(paths, variable).tolist()]
        self.save()

    def select(self, index=None, all=True, query=None, **additional_kwargs):
        if index is not None:
            for i, row in self.df.iloc[index]:
                yield dfseries2dct(row)
        elif query is not None:
            qdf = self.df.query(query)
            for i, row in qdf.iterrows():
                yield dfseries2dct(row)
        elif all is True:
            for i, row in self.df.iterrows(**additional_kwargs):
                yield dfseries2dct(row)

    def refresh(self):
        self.df = self.read(self.filename)

    def save(self, df=None, **pd_kwargs):
        if df is None:
            df = self.df
        df.to_csv(self.filename, **pd_kwargs)

    def read(self, filename, format='pkl'):
        """ pleas install `pandas` """

        if not os.path.isfile(filename):
            return self.pd.DataFrame()
        if self.format == 'pkl':
            df = self.pd.read_pickle(self.filename)
        elif self.format == 'csv':
            df = self.pd.read_csv(self.filename, error_bad_lines=False)
        else:
            raise NotImplementedError('Format %s not supported, '
                                      'use `.csv` or `.pkl`' % format)
        return df


def blob(array):
    """Convert array to blob/buffer object."""

    if array is None:
        return None
    if len(array) == 0:
        array = np.zeros(0)
    if array.dtype == np.int64:
        array = array.astype(np.int32)
    if not np.little_endian:
        array = array.byteswap()
    return memoryview(np.ascontiguousarray(array))


def deblob(buf, dtype=float, shape=None):
    """Convert blob/buffer object to ndarray of correct dtype and shape.
    (without creating an extra view)."""
    if buf is None:
        return None
    if len(buf) == 0:
        array = np.zeros(0, dtype)
    else:
        array = np.frombuffer(buf, dtype=dtype)
        if not np.little_endian:
            array = array.byteswap()
    if shape is not None:
        array.shape = shape
    return array


def encode(obj, binary=False):
    if binary:
        return object_to_bytes(obj)
    return ase.io.jsonio.encode(obj)


def decode(txt, lazy=False):
    if lazy:
        return txt
    if isinstance(txt, str):
        return ase.io.jsonio.decode(txt)
    return bytes_to_object(txt)


class AtomsData:
    """
    First write atoms and return of that ID
    write descriptor, write the source_ID

    """
    atomsdata_parameters = {
        'filename': {'default': "'atoms.db'", 'assert': isStr},
        'tables': {'default': 'None', 'assert': 'True'},
        'static_table_list': {'default': 'None', 'assert': 'True'},
        'search_table_list': {'default': 'None', 'assert': 'True'},
        'calculation_table_list': {'default': 'None', 'assert': 'True'},
        'initialized': {'default': 'None', 'assert': isbool},
        'sbdesc': {'default': 'None', 'assert': 'True'}
    }

    tables = OrderedDict(
        meta=OrderedDict(
            ctimeout='real'
        ),
        invariants=OrderedDict(
            symbols='text',
            atoms_number='integer',
            cell='blob',
            shape='blob'
        ),
        atoms=OrderedDict(
            positions='blob',
            label='text',
            status='text',
            start_time='real',
            potential='real',
            potentials='blob',
            forces='blob',
            finish_time='real'
        ),
        sbdesc=OrderedDict(
            symbol='text',
            desc='blob',
            positions='blob',
            idx='integer',
            source_id='integer',
            potential='real',
            force='blob'
        ),
        prjdesc=OrderedDict(
            coord='blob',
            label='text',
            status='text',
            start_time='real',
            potential='real',
            grad='blob',
            finish_time='real',
            positions='blob',
            forces='blob',
        ),
    )
    static_table_list = ['meta', 'invariants']
    search_table_list = ['atoms']
    calculation_table_list = ['atoms']

    def __init__(self, filename='descriptor.db', tables=None,
                 search_table_list=None, calculation_table_list=None,
                 static_table_list=None, **kwargs):
        self.filename = filename
        self.tables = tables or self.tables
        self.static_table_list = static_table_list or self.static_table_list
        self.search_table_list = search_table_list or self.search_table_list
        self.calculation_table_list = calculation_table_list or \
            self.calculation_table_list
        self.create_tables()
        self.static_tables()
        if 'sbdesc' in self.calculation_table_list:
            self.sbdesc_settings(**kwargs)
        if 'prjdesc' in self.calculation_table_list:
            self.prjdesc_settings(**kwargs)

    def create_tables(self, tables=None):
        tables = tables or self.tables
        directory, prefix = os.path.split(self.filename)
        if directory == '':
            directory = '.'
        if not os.path.exists(directory):
            os.makedirs(directory)
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for table_name, entries in tables.items():
            create_statement = 'CREATE TABLE ' + table_name
            if table_name in self.search_table_list:
                entry_statement = ' ID INTEGER PRIMARY KEY AUTOINCREMENT, '
            else:
                entry_statement = ''
            for key, dtype in entries.items():
                entry_statement += '%s %s, ' % (key, dtype)
            create_statement += ' ({})'.format(entry_statement[:-2])
            try:
                c.execute(create_statement)
            except OperationalError:
                continue
        conn.commit()
        conn.close()

    def static_tables(self, static_table_list=None):
        """
        Purpose of this table is to contain info that rarely changes, and

        """
        static_table_list = static_table_list or self.static_table_list
        # conn = sqlite3.connect(self.filename)
        # c = conn.cursor()
        # conn.commit()
        # conn.close()
        self.atomsdata_parameters['ctimeout'] = {'default': 'None',
                                                 'assert': 'True'}
        self.ctimeout = 3600.
        self.atomsdata_parameters['positions_shape'] = {'default': 'None',
                                                        'assert': 'True'}
        # self.positions_shape = (22, 3)

    def sbdesc_settings(self, **kwargs):
        self.sbdesc = SphericalHarmonicDescriptor()

    def prjdesc_settings(self, prjdesc=None, **kwargs):
        self.prjdesc = prjdesc

    def read(self, ids, tables=None, columns=None):
        """
        ids: {atoms: [1, 2, 3, ..], descriptor: [2, 3, ..], ...}
        return {atoms: [(data[0], ..., ), ()]}
        """
        tables = tables or self.tables
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        data = {}
        for table_name, entries in tables.items():
            id = ids.get(table_name)
            if id is None:
                continue
            data[table_name] = []
            column = entries.keys()
            select_statement = "SELECT " + ', '.join(column) + " FROM "
            select_statement += table_name + " WHERE rowid="
            for i in id:
                c.execute(select_statement + str(i))
                data[table_name].append(c.fetchone())
        conn.commit()
        conn.close()
        return self.czvf(data, xzvf=True)

    def write(self, data, ids=None, tables=None):
        tables = tables or self.tables
        data = self.czvf(data, tables=tables)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        ids = {}
        for table_name, entries in tables.items():
            datum = data.get(table_name)
            if datum is None:
                continue
            c.execute('SELECT max(ROWID) from ' + table_name)
            maxid = c.fetchone()[0] or 0
            insert_statement = 'INSERT INTO ' + table_name
            insert_statement += ' ({})'.format(', '.join(entries.keys()))
            insert_statement += ' VALUES '
            insert_statement += '({})'.format(', '.join(['?'] * len(entries)))
            c.executemany(insert_statement, datum)
            id_statement = "SELECT ROWID FROM " + table_name
            id_statement += " WHERE ROWID > %d ORDER BY ROWID ASC" % maxid
            c.execute(id_statement)
            ids[table_name] = [id[0] for id in c.fetchall()]
        conn.commit()
        conn.close()
        return ids

    def update(self, ids, data, tables=None):
        """
        Basically
        """
        tables = tables or self.tables
        data = self.czvf(data, tables=tables)
        # self.delete(ids)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for table_name, entries in tables.items():
            datum = data.get(table_name)
            if datum is None:
                continue
            id = ids[table_name]
            dat_list_with_id = [(*dat, i) for dat, i in zip(datum, id)]
            update_statement = 'UPDATE ' + table_name + ' SET '
            update_statement += '=?, '.join(entries.keys()) + '=? '
            update_statement += 'WHERE rowid=?'
            c.executemany(update_statement, dat_list_with_id)
        conn.commit()
        conn.close()

    def delete(self, ids, tables=None):
        # Currently only supports deleting all informations of specific id
        """
        ids : dict
            ids = {'atoms': [3, 4, 5], 'sbdesc': [2, 3, 4, 5]}
        """
        tables = tables or self.tables
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for table_name, entries in tables.items():
            if ids.get(table_name) is None:
                continue
            delete_statement = 'DELETE FROM ' + table_name + ' WHERE id in '
            delete_id_list = ', '.join([str(id) for id in ids[table_name]])
            delete_statement += '({})'.format(delete_id_list)
            c.execute(delete_statement)
        conn.commit()
        conn.close()

    def search(self, paths, coords, search_table_list=None, **kwargs):
        """
        Search perfect match exist,
        if not, distance calculate
        """
        stl = search_table_list or self.search_table_list
        ids = {}
        for table_name in stl:
            input_dict = self.get_input_dict(paths, coords,
                                             search_table_list=[table_name],
                                             **kwargs)
            _search = getattr(self, '_search_%s' % table_name)
            ids[table_name] = _search(paths, input_dict[table_name], **kwargs)
            kwargs['source_id'] = ids
        return ids

    def _search_atoms(self, paths, positions_list, pack_null=False, **kwargs):
        """
        Only search atoms table positions_arr exists.
        positions_arr : M x N x 3 array where M is the number of data and N is
                        the number of atoms
        If pack null given, fill up the void into slot where no data found.
        Search perfect match exist,
        if not, check similar structure, or lower distance calculate
        """
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        if type(positions_list) != list:
            positions_list = [positions_list]
        # coords = np.atleast_3d(coords)
        ids = []
        for positions in positions_list:
            # Check Perfect match
            select_statement = "SELECT rowid FROM atoms WHERE "
            select_statement += "positions=?"
            c.execute(select_statement, [blob(positions)])
            id = c.fetchone()
            # Check Similar match
            if id is None:
                pass
            # PAD empty slots
            if id is None and pack_null:
                data = {}
                data['atoms'] = self._create_atoms_data(paths, [positions],
                                                        pack_null=True,
                                                        **kwargs)
                id = self.write(data)['atoms']
            ids.extend(id)
        conn.commit()
        conn.close()
        return ids

    def _search_sbdesc(self, paths, sbdesc_list, pack_null=False, **kwargs):
        """
        Only search for descriptor table in symbol for descriptor exists.
        sbdesc_list : list of tuples contain ('symbol', 'descriptor',
                                            'positions, 'idx', 'source_id')
        """
        if type(sbdesc_list) != list:
            sbdesc_list = list(sbdesc_list)
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        ids = []
        for sbdesc in sbdesc_list:
            symbol, d = sbdesc[:2]
            # Check Perfect match
            select_statement = "SELECT rowid FROM sbdesc WHERE "
            select_statement += "symbol=? AND desc=?"
            # select_statement += ", ({0}, {1})".format(symbol, blob(d))
            c.execute(select_statement, (symbol, blob(d)))
            id = c.fetchone()
            # Check Similar match
            if id is None:
                pass
            # PAD empty slots
            if id is None and pack_null:
                data = {}
                data['sbdesc'] = self._create_sbdesc_data(
                    paths, [sbdesc], pack_null=True, **kwargs)
                id = self.write(data)['sbdesc']

            ids.extend(id)
        conn.commit()
        conn.close()
        return ids

    def _search_prjdesc(self, paths, coords, pack_null=False,
                        prjdesc_search_similar=False, prjdesc_similar_tol=None,
                        **kwargs):
        """
        Only search atoms table positions_arr exists.
        coords : dim x A x N array where N is the number of data and A is
                        the number of atoms
        If pack null given, fill up the void into slot where no data found.
        Search perfect match exist,
        if not, check similar structure, or lower distance calculate
        """
        tol = prjdesc_similar_tol
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        coords = np.atleast_3d(coords)
        # coords = np.atleast_3d(coords)
        N = coords.shape[-1]
        ids = []
        for i in range(N):
            coord = coords[..., i]
            # Check Perfect match
            select_statement = "SELECT rowid FROM atoms WHERE "
            select_statement += "coord=?"
            c.execute(select_statement, [blob(coord)])
            id = c.fetchone()
            # Check Similar match
            if id is None and prjdesc_search_similar:
                if i == 0:
                    data_prj = self.read_all_data(tables=['prjdesc'])['prjdesc']
                    coords_data = np.array([d[1] for d in data_prj]).T
                distances = np.dot(coord, coords_data)
                if np.abs(distances.min() - np.linalg.norm(coord)) < tol:
                    id = data_prj[np.argmin(distances)]['rawid']

            # PAD empty slots
            if id is None and pack_null:
                data = {}
                data['prjdesc'] = self._create_prjdesc_data(paths, [coord],
                                                            pack_null=True,
                                                            **kwargs)
                id = self.write(data)['prjdesc']
            ids.extend(id)
        conn.commit()
        conn.close()
        return ids

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
              arr_dict['atoms'] = [np.array([...]), ..., np.array([...])]
              arr_dict['descriptor']
                  = { 'H': [np.array([...]), ..., np.array([...])],
                      'C': [...],
                      ...}
        """
        # @@@@@@@@@@@@@@@@ Temporal Code >>>
        self.positions_shape = (paths.M, paths.D)
        self.potentials_shape = (paths.A)
        self.forces_shape = (paths.M, paths.D)
        self.desc_shape = (15)
        self.force_shape = (paths.D)
        # ############### Temproal end <<<

        coords = np.atleast_3d(coords)
        ids = self.search(paths, coords, pack_null=True, **kwargs)
        print(ids)
        while True:
            # Not existing data + Failed data
            ids_ntbc = self.get_ids_need_to_be_calculated(ids)
            arr_dict_ntbc = self.get_arr_dict_need_to_be_calculated(ids_ntbc)
            # you_need_calculation = self.check_calculation_is_necessary(ids)
            try:
                self.queue(ids_ntbc, arr_dict_ntbc, **kwargs)
                data = self.create_data(paths, arr_dict_ntbc, ids=ids_ntbc,
                                        update_status=True, update_inloop=True,
                                        **kwargs)
            except Exception as e:
                self.queue(ids_ntbc, arr_dict_ntbc, status='Failed', **kwargs)
                raise Exception(str(e)) from e
            self.update(ids_ntbc, data, **kwargs)
            if self.You_and_I_have_unfinished_business(ids, **kwargs):
                time.sleep(5)
            else:
                break
        if return_data:
            return self.read(ids)
        return ids

    def create_data(self, paths, arr_dict, data_table_list=None,
                    update_inloop=False, **kwargs):
        """
        This function only creates the data based on the given `arr_dict`
        arr_dict : dict contains pathways to be calculated
              arr_dict['atoms'] = [np.array([...]), ..., np.array([...])]
              arr_dict['sbdesc'] = [(symbol, desc, positions, idx, source_id),
                                     ..., ]
        Datum : tuple {descriptor, atom_idx, E, F}
        Index : coords ->
        return : X -, Y -, Paths-, coords-; E, F
        """
        dtl = data_table_list or self.tables.keys()
        data = {}
        for table_name in dtl:
            if arr_dict.get(table_name) is None:
                continue
            _create_data = getattr(self, '_create_%s_data' % table_name)
            data[table_name] = _create_data(paths, arr_dict[table_name],
                                            **kwargs)
            if update_inloop:
                ids_name = kwargs.get('ids')[table_name]
                if ids_name is None:
                    pass
                else:
                    ids_dict = {table_name: ids_name}
                    self.update(ids_dict, {table_name: data[table_name]})

        # NxP sets of tuple or Nx1 sets of tuple
        return data

    def _create_meta_data(self, paths, coords, ctimeout=3600, **kwargs):
        """
        'metadata': OrderedDict(
            ctimeout='real',
            mtime='real'
            ),
        """
        metadata = OrderedDict(ctimeout=ctimeout)
        return metadata

    def _create_atoms_data(self, paths, positions_list, pack_null=False,
                           ids=None, update_status=False, **kwargs):
        """
        positions_list : list of arrays
        update_status : list of ids
        return : list of tuples containing
              (positions, label, status, start_time,
                          potential, potentials, forces, finish_time)
        """
        # @@@@@@@@@@@@ Temp, itshould generate automatically
        props = kwargs.get('properties') or \
                    paths.real_model.implemented_properties
        # @@@@@@@@@@@@ Temp
        if ids is not None:
            ids_atoms = ids['atoms']
        else:
            ids_atoms = [None] * len(positions_list)
        if type(positions_list) != list:
            positions_list = list(positions_list)
        atoms_data = []
        for id, positions in zip(ids_atoms, positions_list):
            label = None
            status = None
            start_time = None
            potential = None
            forces = None
            potentials = None
            finish_time = None
            if pack_null:
                pass
            else:
                positions_T = positions.T  # N x 3 -> 3xN
                coords = np.atleast_3d(positions_T)  # 3xNx1
                start_time = time.time()
                status = 'Running'
                if update_status:
                    datum = (positions, label, status, start_time,
                             potential, potentials, forces, finish_time)
                    if id is None:
                        raise NotImplementedError('To update status, need ids')
                    self.update({'atoms': [id]}, {'atoms': [datum]})

                results = paths.get_properties(coords=coords, properties=props,
                                               real_model=True, caching=True)
                finish_time = time.time()
                # @@@@@@@ It should handle property dynamically
                label = results.get('label')
                positions = results.get('positions') or positions
                potentials = results.get('potentials')
                forces = results['forces']
                potential = results['potential'][0]
                # @@@@@@@ It should handle proeprty dynamically
                status = 'Finished'
            atoms_data.append((positions, label, status, start_time,
                               potential, potentials, forces, finish_time))
        return atoms_data

    def _create_sbdesc_data(self, paths, sbdesc_list, pack_null=False,
                            source_id=None, **kwargs):
        """
        sbdesc_list : list of tuples containing
          ('symbol', 'desc', 'positions', 'idx', 'source_id')
          or
        return : list of tuples containing
          ('symbol', 'desc', 'positions', 'idx', 'source_id', 'potential')
        """
        s2i = {}
        data_atoms = {}
        for sbdesc in sbdesc_list:
            symbol, desc, positions, idx, source_id = sbdesc
            if s2i.get(source_id) is None:
                s2i[source_id] = []
            s2i[source_id].append(idx)
        n2i = self.name2idx['atoms']
        sbdesc_data = []
        for sbdesc in sbdesc_list:
            symbol, desc, positions, idx, source_id = sbdesc
            potential = None
            force = None
            if pack_null:
                pass
            else:
                # Search source ID that potentials exists.
                # If not, calculate directly

                data_atoms = self.read({'atoms': [source_id]})['atoms']
                potentials = data_atoms[0][n2i['potentials']]
                forces = data_atoms[0][n2i['forces']]
                if potentials is None:
                    raise AttributeError("No potentials found %d" % source_id)
                potential = potentials[idx]
                force = forces[idx]

            sbdesc_data.append((symbol, desc, positions, idx, source_id,
                                potential, force))
        return sbdesc_data

    def _create_prjdesc_data(self, paths, coords, pack_null=False,
                             ids=None, update_status=False, **kwargs):
        """
        positions_list : list of arrays
        update_status : list of ids
        return : list of tuples containing
              (positions, label, status, start_time,
                          potential, potentials, forces, finish_time)
        prjdesc=OrderedDict(
            coord='blob',
            label='text',
            status='text',
            start_time='real',
            potential='real',
            grad='blob',
            finish_time='real',
            positions='blob',
            forces='blob',
        ),
        """
        # @@@@@@@@@@@@ Temp, itshould generate automatically
        props = kwargs.get('properties') or \
                        paths.real_model.implemented_properties
        # @@@@@@@@@@@@ Temp
        coords = np.atleast_3d(coords)
        N = coords.shape[-1]
        if ids is not None:
            ids_prjdesc = ids.get('prjdesc')
        else:
            ids_prjdesc = [None] * N

        prjdesc_data = []
        for i, id in enumerate(ids_prjdesc):
            coord = coords[..., i, np.newaxis]
            label = None
            status = None
            start_time = None
            potential = None
            grad = None
            finish_time = None
            forces = None
            if pack_null:
                pass
            else:
                start_time = time.time()
                status = 'Running'
                if update_status:
                    datum = (coord, label, status, start_time,
                             potential, grad, forces, finish_time)
                    if id is None:
                        raise NotImplementedError('To update status, need ids')
                    self.update({'atoms': [id]}, {'atoms': [datum]})

                results = paths.get_properties(coords=coord, properties=props,
                                               real_model=True, caching=True)
                finish_time = time.time()
                # @@@@@@@ It should handle property dynamically
                label = results.get('label')
                positions = results.get('positions') or positions
                potentials = results.get('potentials')
                forces = results['forces']
                potential = results['potential'][0]
                # @@@@@@@ It should handle proeprty dynamically
                status = 'Finished'
            prjdesc_data.append((positions, label, status, start_time,
                                 potential, potentials, forces, finish_time))
        return prjdesc_data

    def get_input_dict(self, paths, coords, search_table_list=None,
                       **kwargs):
        """
        Using coords, construct suitable input format for search table
         'atoms' -> positions_list : M x N
         'descriptor' -> list of tuples contain
          ('symbol', 'descriptor', 'positions', 'idx', 'source_id')
        """
        stl = search_table_list or self.search_table_list
        input_dict = {}
        if 'atoms' in stl:
            positions_arr = np.atleast_3d(coords).T
            input_dict['atoms'] = [positions for positions in positions_arr]
        if 'sbdesc' in stl:
            positions_arr = np.atleast_3d(coords).T
            source_id = kwargs.get('source_id', {})
            source_id = source_id.get('atoms', [None] * len(positions_arr))
            symbols = paths.symbols
            sbdescriptors = self.sbdesc(paths, coords)
            input_dict['sbdesc'] = []
            for i in range(len(positions_arr)):
                sbdescriptor, pos = sbdescriptors[i], positions_arr[i]
                sc_id = source_id[i]
                for j in range(len(pos)):
                    sym = symbols[j]
                    desc = sbdescriptor[j]
                    idx = j
                    input_dict['sbdesc'].append((sym, desc, pos, idx, sc_id))
        return input_dict

    def get_arr_dict_need_to_be_calculated(self, ids,
                                           calculation_table_list=None):
        ctl = calculation_table_list or self.calculation_table_list
        arr_ntbc = OrderedDict()
        # For readable code, indexing with string. Not number
        name2idx = self.name2idx
        if 'atoms' in ctl:
            arr_ntbc['atoms'] = []
            data_atoms = self.read({'atoms': ids['atoms']})['atoms']
            n2i = name2idx['atoms']
            for data in data_atoms:
                positions = data[n2i['positions']]
                arr_ntbc['atoms'].append(positions)
        if 'sbdesc' in ctl:
            arr_ntbc['sbdesc'] = []
            data_sbdesc = self.read({'sbdesc': ids['sbdesc']})['sbdesc']
            n2i = name2idx['sbdesc']
            for data in data_sbdesc:
                symbol, desc, positions, idx, source_id = data[:5]
                arr_ntbc['sbdesc'].append((symbol, desc, positions, idx,
                                           source_id))
        return arr_ntbc

    def get_ids_need_to_be_calculated(self, ids, calculation_table_list=None):
        """
        Given ids that spans all ids, check if calculation is necessary.
        if it is true
        First, check the status is None
        ids : dictionary of lists that contains id
               {'atoms': [1, 2, ..], 'sbdesc': [1, 2, ..], ...}

        """
        ctl = calculation_table_list or self.calculation_table_list
        # For readable code, indexing with string. Not number
        name2idx = self.name2idx
        ids_ntbc = OrderedDict()
        if 'atoms' in ctl:
            n2i = name2idx['atoms']
            ids_ntbc['atoms'] = []
            atoms_data = self.read({'atoms': ids['atoms']})['atoms']
            # data : list, datum : tuple
            for id, datum in zip(ids['atoms'], atoms_data):
                status = datum[n2i['status']]
                if status is None or status == 'Failed':
                    ids_ntbc['atoms'].append(id)
                elif status == 'Finished':
                    continue
                elif status == 'Running':
                    ctimeout = self.ctimeout
                    # Time out
                    if time.time() - datum[n2i['start_time']] > ctimeout:
                        ids_ntbc['atoms'].append(id)
                    else:
                        continue
                else:
                    raise NotImplementedError('Can not reconize %s' % status)
        if 'sbdesc' in ctl:
            n2i = name2idx['sbdesc']
            ids_ntbc['sbdesc'] = []
            ids_sbdesc = ids['sbdesc']
            data_sbdesc = self.read({'sbdesc': ids_sbdesc})['sbdesc']
            # source_ids : dictionary ordered by source_id
            #    For example, {source_id: [desc_id_1, desc_id_2, ...]}
            source_ids = dict()
            for id, data in zip(ids_sbdesc, data_sbdesc):
                source_id = data[n2i['source_id']]
                if source_ids.get(source_id) is None:
                    source_ids[source_id] = []
                source_ids[source_id].append(id)
            ids_atoms = source_ids.keys()
            data_atoms = self.read({'atoms': ids_atoms})['atoms']
            n2i = name2idx['atoms']
            for id, data in zip(ids_atoms, data_atoms):
                status = data[n2i['status']]
                if status is None or status == 'Failed':
                    ids_ntbc['sbdesc'].extend(source_ids[id])
                elif status == 'Finished':
                    continue
                elif status == 'Running':
                    ctimeout = self.ctimeout
                    start_time = data[n2i['start_time']]
                    # Time out
                    if time.time() - start_time > ctimeout:
                        ids_ntbc['sbdesc'].extend(source_ids[id])
                    else:
                        continue
                else:
                    raise NotImplementedError('Can not recognize')
        return ids_ntbc

    def queue(self, ids, arr_dict, calculation_table_list=None,
              status='Running', **kwargs):
        ctl = calculation_table_list or self.calculation_table_list
        if 'atoms' in ctl:
            ids_atoms = ids['atoms']
            positions_list = arr_dict['atoms']
            for id, positions in zip(ids_atoms, positions_list):
                label = None
                status = status
                potential = None
                forces = None
                potentials = None
                finish_time = None
                start_time = time.time()
                datum = (positions, label, status, start_time,
                         potential, potentials, forces, finish_time)
                self.update({'atoms': [id]}, {'atoms': [datum]})

    def You_and_I_have_unfinished_business(self, ids,
                                           calculation_table_list=None):
        """
        Kill Bill
        Check calculation of given ids are finished
        """
        ctl = calculation_table_list or self.calculation_table_list
        name2idx = self.name2idx
        intrim_report = []
        if 'atoms' in ctl:
            n2i = name2idx['atoms']
            atoms_data = self.read({'atoms': ids['atoms']})['atoms']
            statuses = np.array([datum[n2i['status']] for datum in atoms_data])
            if np.all(statuses == 'Finished'):
                intrim_report.append(False)
            else:
                return True

        if 'sbdesc' in ctl:
            n2i = name2idx['sbdesc']
            # source_ids = [id for id in ids['sbdesc']]
            data = self.read({'sbdesc': ids['sbdesc']})['sbdesc']
            ids_atoms = []
            for datum in data:
                source_id = datum[n2i['source_id']]
                if source_id in ids_atoms:
                    continue
                ids_atoms.append(source_id)
            n2i = name2idx['atoms']
            atoms_data = self.read({'atoms': ids_atoms})['atoms']
            statuses = np.array([datum[n2i['status']] for datum in atoms_data])
            if np.all(statuses == 'Finished'):
                intrim_report.append(False)
            else:
                return True
        return np.any(intrim_report)

    def czvf(self, data, tables=None, xzvf=False):
        """
        data : dict(list of tuples)
        return : data_dict(list of tuples)
        """
        tables = tables or self.tables
        data_dict = OrderedDict()
        name2idx = self.name2idx
        for table_name, entries in tables.items():
            if data.get(table_name) is None:
                continue
            data_dict[table_name] = []
            # list of tuples
            data_in_table = data[table_name]
            # dtypes = [value for key, value in tables[table_name].items()]
            n2i = name2idx[table_name]
            for datum in data_in_table:
                # tuples
                cache = list(datum)
                for column, dtype in entries.items():
                    if dtype == 'blob':
                        i = n2i[column]
                        if xzvf:
                            shape = getattr(self, column + '_shape')
                            cache[i] = deblob(datum[i], shape=shape)
                        else:
                            cache[i] = blob(datum[i])
                data_dict[table_name].append(tuple(cache))
        return data_dict

    def dict2tuple(self, dict_data, tables=None):
        tables = tables or self.tables
        data = OrderedDict()
        for table_name, entries in tables.items():
            # dict_datum : list of dictionary
            dict_datum = dict_data.get(table_name)
            if dict_datum is None:
                continue
            data[table_name] = []
            # dict_dat: dictionary
            for dict_dat in dict_datum:
                dat = []
                for key, value in dict_dat.items():
                    dat.append(value)
                data[table_name].append(tuple(dict_dat))
        return data

    def tuple2dict(self, data, tables=None):
        tables = tables or self.tables
        idx2name = self.idx2name
        dict_data = OrderedDict()
        for table_name, entries in tables.items():
            # datum : list of tuples
            datum = data.get(table_name)
            if datum is None:
                continue
            dict_data[table_name] = []
            i2n = idx2name[table_name]
            # dat: tuples
            for dat in datum:
                dict_dat = OrderedDict()
                for i, d in enumerate(dat):
                    dict_dat[i2n[i]] = d
                dict_data[table_name].append(dict_dat)
        return dict_data

    @property
    def name2idx(self):
        tables = self.tables
        if getattr(self, '_name2idx', None) is not None:
            return self._name2idx
        name2idx = {}
        idx2name = {}
        for table_name, entries in tables.items():
            name2idx[table_name] = OrderedDict()
            idx2name[table_name] = OrderedDict()
            for idx, name in enumerate(entries.keys()):
                name2idx[table_name][name] = idx
                idx2name[table_name][idx] = name
        self._name2idx = name2idx
        self._idx2name = idx2name
        return self._name2idx

    @property
    def idx2name(self):
        tables = self.tables
        if getattr(self, '_idx2name', None) is not None:
            return self._idx2name
        name2idx = {}
        idx2name = {}
        for table_name, entries in tables.items():
            name2idx[table_name] = OrderedDict()
            idx2name[table_name] = OrderedDict()
            for idx, name in enumerate(entries.keys()):
                name2idx[table_name][name] = idx
                idx2name[table_name][idx] = name
        self._name2idx = name2idx
        self._idx2name = idx2name
        return self._idx2name


class PickledData:
    invar_list = ['symbols', 'dct']
    meta_list = ['dcut', 'population_size', 'iteration']
    var_list = ['coords', 'finder_results', 'parents', 'id',
                'atomsdata_database']

    def __init__(self, filename):
        self.filename = filename
        self.lock = Lock(filename + '.lock')
        self.lock.timeout = 5
        self.meta_data = {}
        self.invariants = {}
        self.variables = {}
        dir = os.path.dirname(self.filename)
        if dir == '':
            dir = '.'
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(self.filename):
            self.save()
        else:
            self.time_stamp = time.time()
            self.refresh()
            self.load_invariants()

    def __getitem__(self, key):
        self.refresh()
        symbols = self.invariants['symbols']
        dct = self.invariants['dct']
        coords = self.variables['coords'][key]
        dct['results'] = self.variables['results'][key].copy()
        dct['id'] = self.variables['id'][key]
        dct['parents'] = self.variables['parents'][key]
        return symbols, coords, dct

    def __getattr__(self, key):
        if key in ['filename', 'meta_data', 'invariants', 'variables',
                   'time_stamp', 'lock']:
            return super().__getattribute__(key)
        self.refresh()
        return self.meta_data[key]

    def __setattr__(self, key, value):
        if key in ['filename', 'meta_data', 'time_stamp',
                   'invariants', 'variables', 'lock']:
            super().__setattr__(key, value)
        else:
            self.meta_data.update({key: value})
            self.save()

    @property
    def iteration(self):
        return self.meta_data.get('iteration', 0)

    @iteration.setter
    def iteration(self, iteration):
        self.lock.acquire()
        self.meta_data['iteration'] = iteration
        self.save()
        self.lock.release()

    def replace(self, paths=None, id=None):
        for variable in self.var_list:
            value = getattr(paths, variable)
            self.variables[variable][id] = value

    def refresh(self):
        f = open(self.filename, 'rb')
        if self.time_stamp == pickle.load(f):
            pass
        else:
            self.meta_data = pickle.load(f)
            self.variables = pickle.load(f)
            if self.variables.get('parents') is not None:
                parents = self.variables.get('parents')
                self.variables['parents'] = parents.astype(object)
        f.close()

    def load_invariants(self):
        with open(self.filename, 'rb') as f:
            pickle.load(f)
            pickle.load(f)
            pickle.load(f)
            self.invariants = pickle.load(f)

    def save(self):
        self.time_stamp = time.time()
        with open(self.filename, 'wb') as f:
            pickle.dump(self.time_stamp, f)
            pickle.dump(self.meta_data, f)
            pickle.dump(self.variables, f)
            pickle.dump(self.invariants, f)
