import os
import time
import pickle
import sqlite3
import numpy as np
from sqlite3 import OperationalError
from collections import OrderedDict
from scipy.spatial import KDTree

import ase.io.jsonio
from ase.db.core import bytes_to_object, object_to_bytes

from ase.pathway.utils import isStr, isbool
from ase.pathway.descriptor import SphericalHarmonicDescriptor


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


class PathsData:
    """
    Current
      Save : Paths ->  pickle serialize -> sqlite
      load : Sqlite -> pickle serialize -> Paths
    Future plan
      Save : Paths -> Pathsrow(ver ctrl) -> dct -> json serialize -> sqlite
      load : Sqlite -> json serialize -> dct -> Pathsrow(ver ctrl) -> Paths
    """
    metadata = {
        'ctime': 'real'
    }
    invariants = {
        'symbols': 'text',
    }
    variables = dict(
        paths='blob',
        **dict(zip(['rel%d' % i for i in range(5)], ['real'] * 5)),
        **dict(zip(['txt%d' % i for i in range(5)], ['text'] * 5)),
        **dict(zip(['blb%d' % i for i in range(5)], ['blob'] * 5)),
        **dict(zip(['shp%d' % i for i in range(5)], ['blob'] * 5)),
        **dict(zip(['int%d' % i for i in range(5)], ['int'] * 5))
    )
    key_mapper = {'blb0': 'coords', 'coords': 'blb0'}

    create_table_list = ['metadata', 'invariants', 'variables']
    target_variables = ['symbols', 'coords']

    def __init__(self, filename='pathsdata.db', metadata=None, invariants=None,
                 variables=None, key_mapper=None, **kwargs):
        self.filename = filename
        self.metadata = metadata or self.metadata
        self.invariants = invariants or self.invariants
        self.variables = variables or self.variables
        self.key_mapper = key_mapper or self.key_mapper
        self.create_tables(**kwargs)

    def create_tables(self, create_table_list=None):
        create_table_list = create_table_list or self.create_table_list
        directory, prefix = os.path.split(self.filename)
        if directory == '':
            directory = '.'
        if not os.path.exists(directory):
            os.makedirs(directory)
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for table_name in create_table_list:
            create_statement = 'CREATE TABLE IF NOT EXISTS ' + table_name
            entry_statement = 'ID INTEGER PRIMARY KEY AUTOINCREMENT, '
            entries = getattr(self, table_name)
            for key, dtype in entries.items():
                entry_statement += '%s %s, ' % (key, dtype)
            create_statement += ' ({})'.format(entry_statement[:-2])
            c.execute(create_statement)
        conn.commit()
        conn.close()

    def recreate_tables(self):
        """
        Drop table
        """
        pass

    def read(self, ids=None, query=None, columns=None, key_mapper=None,
             where=" WHERE ", table_name='variables'):
        """
        ids : list of int
        query : 'rowid=1' or 'rowid IN (1, 2, 3, 4)' ...
        columns = list of str, ['init', 'fin', ... ]
        return [paths1, paths2, ...]
        """
        if query is None and ids is None:
            return None
        key_mapper = key_mapper or self.key_mapper
        columns = columns or ['paths']
        columns = [key_mapper.get(column, column) for column in columns]
        if query is not None:
            for mapper in key_mapper.items():
                query = query.replace(*mapper)
        if ids is not None:
            query = 'rowid IN ({})'.format(', '.join([str(id) for id in ids]))

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        select_statement = "SELECT " + ', '.join(columns) + " FROM "
        select_statement += table_name + where + query
        c.execute(select_statement)
        data = c.fetchall()
        conn.commit()
        conn.close()
        return self.decode(data, columns, key_mapper=key_mapper)

    def write(self, data=None, key_mapper=None, table_name='variables'):
        """
        data = [{'coords': arr, 'paths': Paths()}, ]
        """
        if data is None or data == []:
            return None
        key_mapper = key_mapper or self.key_mapper
        data = self.encode(data)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        ids = []
        for datum in data:
            columns, dat = zip(*datum.items())
            columns = [key_mapper.get(c, c) for c in columns]
            insert_statement = 'INSERT INTO ' + table_name
            insert_statement += ' ({})'.format(', '.join(columns))
            insert_statement += ' VALUES '
            insert_statement += '({})'.format(', '.join(['?'] * len(columns)))
            c.execute(insert_statement, dat)
            ids.append(c.lastrowid)
        conn.commit()
        conn.close()
        return ids

    def update(self, ids, data=None, key_mapper=None, table_name='variables'):
        """
        ids : list of int; [1, 2, 8, ...]
        data = list of dict; [{'init': arr, 'paths': Paths()}, ...]
        """
        if data is None or data == []:
            return None
        key_mapper = key_mapper or self.key_mapper
        data = self.encode(data)
        # self.delete(ids)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for datum, id in zip(data, ids):
            columns, dat = zip(*datum.items())
            columns = [key_mapper.get(c, c) for c in columns]
            update_statement = 'UPDATE ' + table_name + ' SET '
            update_statement += '=?, '.join(columns) + '=? '
            update_statement += 'WHERE rowid=?'
            c.execute(update_statement, (*dat, id))
        conn.commit()
        conn.close()

    def delete(self, ids):
        # Currently only supports deleting all informations of specific id
        """
        ids : dict
            ids = [1, 2, 3]
        """
        table_name = 'variables'
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        delete_statement = 'DELETE FROM ' + table_name + ' WHERE id in '
        delete_id_list = ', '.join(ids)
        delete_statement += '({})'.format(delete_id_list)
        c.execute(delete_statement)
        conn.commit()
        conn.close()

    def lock(self, timeout=100):
        t = time.time()
        lock = self.filename + '.lock'
        while True:
            if os.path.exists(lock):
                time.sleep(5)
            elif t - time.time() > timeout:
                raise TimeoutError('%s is not reponding' % lock)
            else:
                break

    def release(self):
        if os.path.exists(self.filename + '.lock'):
            os.remove(self.filename + '.lock')

    def count(self, query=None, key_mapper=None):
        if query is None:
            return None
        table_name = 'variables'
        key_mapper = key_mapper or self.key_mapper
        for mapper in key_mapper.items():
            query = query.replace(*mapper)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        select_statement = "SELECT COUNT(*) FROM "
        select_statement += table_name + " WHERE " + query
        c.execute(select_statement)
        count = c.fetchone()
        conn.commit()
        conn.close()
        return count[0]

    def decode(self, data, columns, key_mapper=None):
        """
        columns = ['blb0', 'txt2', 'paths']
        """
        key_mapper = key_mapper or self.key_mapper
        columns = [key_mapper.get(column, column) for column in columns]
        data = [dict(zip(columns, datum)) for datum in data]
        for i, datum in enumerate(data):
            for column in columns:
                if column == 'paths':
                    data[i][column] = pickle.loads(data[i][column])
                elif 'blb' in key_mapper.get(column, column):
                    data[i][column] = deblob(data[i][column], shape=None)
                else:
                    pass
        return data

    def encode(self, data):
        for i, datum in enumerate(data):
            for name, dat in datum.items():
                class_name = dat.__class__.__name__
                if dat is None:
                    pass
                elif class_name == 'Paths':
                    data[i][name] = memoryview(pickle.dumps(dat))
                elif class_name in ['list', 'tuple', 'ndarray']:
                    data[i][name] = blob(np.array(dat))
                elif class_name in ['str', 'int', 'int64', 'float', 'float64']:
                    pass
                else:
                    raise NotImplementedError("Can't encode %s" % class_name)
        return data


class ImageData:
    imgdata_parameters = {
        'filename': {'default': "'image.db'", 'assert': isStr},
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
            image_number='integer',
            cell='blob',
            shape='blob'
        ),
        image=OrderedDict(
            coord='blob',
            label='text',
            status='text',
            start_time='real',
            potential='real',
            potentials='blob',
            gradient='blob',
            finish_time='real',
            positions='blob',
            forces='blob'
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
    )
    static_table_list = ['meta', 'invariants']
    search_table_list = ['image']
    calculation_table_list = ['image']

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

        self._cache = {}

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
        self.imgdata_parameters['ctimeout'] = {'default': 'None',
                                               'assert': 'True'}
        self.ctimeout = 3600.
        self.imgdata_parameters['positions_shape'] = {'default': 'None',
                                                      'assert': 'True'}
        # self.positions_shape = (22, 3)

    def sbdesc_settings(self, **kwargs):
        self.sbdesc = SphericalHarmonicDescriptor()

    def read(self, ids, tables=None, columns={}, **kwargs):
        """
        ids: {image: [1, 2, 3, ..], descriptor: [2, 3, ..], ...}
        return {image: [(data[0], ..., ), ()]}
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
            column = columns.get(table_name) or entries.keys()
            select_statement = "SELECT " + ', '.join(column) + " FROM "
            select_statement += table_name + " WHERE rowid="
            for i in id:
                c.execute(select_statement + str(i))
                data[table_name].append(c.fetchone())
        conn.commit()
        conn.close()
        return self.czvf(data, xzvf=True)

    def read_all(self, tables=None, query='', **kwargs):
        tables = tables or self.tables
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        data = {}
        for table_name, entries in tables.items():
            data[table_name] = []
            select_statement = "SELECT " + ', '.join(entries.keys()) + " FROM "
            select_statement += table_name + query
            c.execute(select_statement)
            data[table_name].append(c.fetchall())
        conn.commit()
        conn.close()
        return self.czvf(data, xzvf=True, tables=tables)

    def write(self, data, ids=None, tables=None, **kwargs):
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

    def update(self, ids, data, tables=None, **kwargs):
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

    def delete(self, ids, tables=None, **kwargs):
        # Currently only supports deleting all informations of specific id
        """
        ids : dict
            ids = {'image': [3, 4, 5], 'sbdesc': [2, 3, 4, 5]}
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

    def _search_image(self, paths, coords, pack_null=False,
                      search_similar_image=True, similar_image_tol=0.1,
                      **kwargs):
        """
        Only search image table positions_arr exists.
        coords: 3 x N x M array where M is the number of data and N is
                        the number of image
        If pack null given, fill up the void into slot where no data found.
        Search perfect match exist,
        if not, check similar structure, or lower distance calculate
        """
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        coords = np.atleast_3d(coords)
        M = coords.shape[-1]
        ids = []
        for m in range(M):
            coord = coords[..., m]
            # Check Perfect match
            select_statement = "SELECT rowid FROM image WHERE "
            select_statement += "coord=?"
            c.execute(select_statement, [blob(coord)])
            id = c.fetchone()
            # Check Similar match
            if id is None and search_similar_image:
                last_id = max(self._cache.get('coords_ids')) or 0
                tol = similar_image_tol
                read_tables = {'image': OrderedDict(coord='blob', rawid='int')}
                new_data = self.read_all(tables=read_tables, last_id=last_id,
                                         query=' WHERE rowid>%d' % last_id
                                         )['image']
                if new_data is None or new_data == []:
                    pass
                else:
                    dim, A, N = *new_data[0].shape, len(new_data)
                    zarr = np.zeros((dim, A, N))
                    new_coords = zarr.copy()
                    new_ids = []
                    for i in range(N):
                        new_coords[..., i] = new_data[i][0]
                        new_ids.append(new_data[i][1])
                    coords_data = self._cache.get('coords_data', zarr.copy())
                    coords_ids = self._cache.get('coords_ids', [])
                    coords_data = np.concatenate([coords_data, new_coords],
                                                 axis=2)
                    coords_ids.extend(new_ids)
                    self._cache['coords_data'] = coords_data
                    self._cache['coords_ids'] = coords_ids
                    distances = np.linalg.norm(coords_data - coord, axis=2)
                    checker = distances < tol
                    # Check similar results exists
                    if np.any(checker):
                        sim_ids = np.array(coords_ids)[checker]
                        n_similar = len(sim_ids)
                        # Check searched id is already exist in model data ids
                        similar_yet_fresh_ids = []
                        for i in range(n_similar):
                            if sim_ids[i] not in paths.model.data_ids['image']:
                                similar_yet_fresh_ids.append(sim_ids[i])
                        # Emergency mode
                        if similar_yet_fresh_ids == []:
                            similar_coords = np.array(coords_data)[..., checker]
                            # Create new coord
                            center = similar_coords.sum(axis=2) / n_similar
                            ce = coord - center
                            e_ce = ce / np.linalg.norm(ce)
                            coord = coord + tol * e_ce
                        else:
                            # pick among fresh ids
                            id = np.random.choice(similar_yet_fresh_ids)
                    else:
                        pass
            # PAD empty slots
            if id is None and pack_null:
                data = {}
                data['image'] = self._create_image_data(paths, coord,
                                                        pack_null=True,
                                                        **kwargs)
                id = self.write(data)['image']
            ids.extend(id)
        conn.commit()
        conn.close()
        return ids

    def _search_sbdesc(self, paths, sbdesc_list, pack_null=False,
                       search_similar_sbdesc=True, similar_sbdesc_tol=0.1,
                       **kwargs):
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
            symbol, desc = sbdesc[:2]
            sym, d = symbol, desc
            # Check Perfect match
            select_statement = "SELECT rowid FROM sbdesc WHERE "
            select_statement += "symbol=? AND desc=?"
            # select_statement += ", ({0}, {1})".format(symbol, blob(d))
            c.execute(select_statement, (sym, blob(d)))
            id = c.fetchone()
            # Check Similar match
            if id is None and search_similar_sbdesc:
                self._cache['sbdesc'] = self._cache.get('sbdesc', {})
                self._cache['sbdesc'][sym] = self._cache['sbdesc'].get(sym, {})
                last_id = 0
                for key, value in self._cache.get('sbdesc', {}).items():
                    _ids = value.get('ids')
                    if _ids is None:
                        continue
                    last_id = max(max(_ids), last_id)
                tol = similar_sbdesc_tol
                read_tables = {'sbdesc': OrderedDict(rawid='int', desc='blob',
                               positions='blob', idx='int', source_id='int')}
                new_data = self.read_all(tables=read_tables, last_id=last_id,
                                         query=' WHERE symbol=%s rowid>%d' %
                                         (symbol, last_id))['sbdesc']
                if new_data is None or new_data == []:
                    cache = self._cache['sbdesc'][symbol]
                    kdtree = cache.get('kdtree')
                    if kdtree is None:
                        exist_similar = False
                    else:
                        dist, _ = kdtree.query_ball_point(d, tol)
                        if _ is None:
                            exist_similar = False
                        else:
                            exist_similar = True
                            similar_ids = cache.get('ids')[_]
                else:
                    N = len(new_data)
                    new_desc = []
                    new_pos, new_ids, new_sid, new_idx = [], [], [], []
                    cell = paths.__dict__.get('cell')
                    for i in range(N):
                        id, desc, pos, idx, sid = new_data[i]
                        new_ids.append(id)
                        new_desc.append(desc)
                        new_pos.append(pos)
                        new_idx.append(idx)
                        new_sid.append(sid)
                        if cell is not None:
                            pos -= pos[idx]
                            basis = np.linalg.solve(cell.T, pos.T).T
                            basis = (basis + 0.5) % 1. - 0.5
                            pos = basis @ cell

                    cache = self._cache['sbdesc'][symbol]
                    cache['ids'] = cache.get('ids', []).extend(new_ids)
                    cache['pos'] = cache.get('pos', []).extend(new_pos)
                    cache['idx'] = cache.get('idx', []).extend(new_idx)
                    cache['sid'] = cache.get('sid', []).extend(new_sid)
                    cache['desc'] = cache.get('desc', []).extend(new_desc)

                    kdtree = KDTree(cache['desc'])
                    _ = kdtree.query(d, tol)
                    if similar_ids is None:
                        exist_similar = False
                    else:
                        exist_similar = True
                        similar_ids = cache.get('ids')[_]

                # Check similar results exists
                if exist_similar:
                    sim_ids = similar_ids
                    n_similar = len(similar_ids)
                    # Check searched id is already exist in model data ids
                    similar_yet_fresh_ids = []
                    for i in range(n_similar):
                        if sim_ids[i] not in paths.model.data_ids['sbdesc']:
                            similar_yet_fresh_ids.append(sim_ids[i])
                    # Emergency mode
                    if similar_yet_fresh_ids == []:
                        pos, idx, sid = sbdesc[2:]
                        sim_pos = np.array(cache['pos'])[_]
                        # Create new coord
                        center = sim_pos.sum(axis=0) / n_similar
                        ce = d - center
                        e_ce = ce / np.linalg.norm(ce)
                        pos = pos + tol * e_ce
                        ##################
                        desc = self.sbdesc(paths, pos.T)
                        sbdesc = (symbol, desc, pos, idx, sid)
                    else:
                        # pick among fresh ids
                        id = np.random.choice(similar_yet_fresh_ids)
            if id is None and pack_null:
                data = {}
                data['sbdesc'] = self._create_sbdesc_data(
                    paths, [sbdesc], pack_null=True, **kwargs)
                id = self.write(data)['sbdesc']

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
              arr_dict['image'] = [np.array([...]), ..., np.array([...])]
              arr_dict['descriptor']
                  = { 'H': [np.array([...]), ..., np.array([...])],
                      'C': [...],
                      ...}
        """
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        self.coord_shape = (paths.D, paths.M)
        self.gradient_shape = (paths.D, paths.M)
        self.potentials_shape = (paths.A)
        self.desc_shape = (15)
        self.positions_shape = (paths.A, 3)
        self.forces_shape = (paths.A, 3)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #

        coords = np.atleast_3d(coords)
        ids = self.search(paths, coords, pack_null=True, **kwargs)
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
              arr_dict['image'] = [np.array([...]), ..., np.array([...])]
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

    def _create_image_data(self, paths, coords, pack_null=False,
                           ids=None, update_status=False, **kwargs):
        """
        coords : dim x A x M or dim x A array
        update_status : list of ids
        return : list of tuples containing
              (coord, label, status, start_time, potential, potentials,
               gradient, finish_time, positions, forces)
        """
        n2i = self.name2idx['image']
        properties = n2i.keys()
        paths.real_model.update_implemented_properties(paths)
        model_properties = paths.real_model.implemented_properties
        props = [p for p in properties if p in model_properties]
        M = coords.shape[-1]
        if ids is not None:
            ids_image = ids['image']
        else:
            ids_image = [None] * M
        image_data = []
        for m in range(M):
            datum = OrderedDict(zip(properties, [None] * len(properties)))
            coord = coords[..., m]
            datum['coord'] = coord
            if pack_null:
                pass
            else:
                datum['start_time'] = time.time()
                datum['status'] = 'Running'
                if update_status:
                    id = ids_image[m]
                    datum_tuple = tuple([val for val in datum.values()])
                    if id is None:
                        raise NotImplementedError('To update status, need ids')
                    self.update({'image': [id]}, {'image': [datum_tuple]})

                results = paths.get_properties(coords=coord, properties=props,
                                               real_model=True, caching=True)
                datum['finish_time'] = time.time()

                for key, val in results.items():
                    datum[key] = val

                datum['status'] = 'Finished'
            image_data.append(tuple([val for val in datum.values()]))
        return image_data

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
        data_image = {}
        for sbdesc in sbdesc_list:
            symbol, desc, positions, idx, source_id = sbdesc
            if s2i.get(source_id) is None:
                s2i[source_id] = []
            s2i[source_id].append(idx)
        n2i = self.name2idx['image']
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

                data_image = self.read({'image': [source_id]})['image']
                potentials = data_image[0][n2i['potentials']]
                forces = data_image[0][n2i['forces']]
                if potentials is None:
                    raise AttributeError("No potentials found %d" % source_id)
                potential = potentials[idx]
                force = forces[idx]

            sbdesc_data.append((symbol, desc, positions, idx, source_id,
                                potential, force))
        return sbdesc_data

    def get_input_dict(self, paths, coords, search_table_list=None,
                       **kwargs):
        """
        Using coords, construct suitable input format for search table
         'image' -> positions_list : M x N
         'descriptor' -> list of tuples contain
          ('symbol', 'descriptor', 'positions', 'idx', 'source_id')
        """
        stl = search_table_list or self.search_table_list
        input_dict = {}
        if 'image' in stl:
            coords = np.atleast_3d(coords)
            input_dict['image'] = coords
        if 'sbdesc' in stl:
            coords = np.atleast_3d(coords)
            positions_arr = np.atleast_3d(coords).T
            source_id = kwargs.get('source_id', {})
            source_id = source_id.get('image', [None] * len(positions_arr))
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
        if 'image' in ctl:
            coord_T_list = []
            data_image = self.read({'image': ids['image']})['image']
            n2i = name2idx['image']
            for data in data_image:
                coord_T_list.append(data[n2i['coord']].T)
            arr_ntbc['image'] = np.array(coord_T_list).T
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
               {'image': [1, 2, ..], 'sbdesc': [1, 2, ..], ...}

        """
        ctl = calculation_table_list or self.calculation_table_list
        # For readable code, indexing with string. Not number
        name2idx = self.name2idx
        ids_ntbc = OrderedDict()
        if 'image' in ctl:
            n2i = name2idx['image']
            ids_ntbc['image'] = []
            image_data = self.read({'image': ids['image']})['image']
            # data : list, datum : tuple
            for id, datum in zip(ids['image'], image_data):
                status = datum[n2i['status']]
                if status is None or status == 'Failed':
                    ids_ntbc['image'].append(id)
                elif status == 'Finished':
                    continue
                elif status == 'Running':
                    ctimeout = self.ctimeout
                    # Time out
                    if time.time() - datum[n2i['start_time']] > ctimeout:
                        ids_ntbc['image'].append(id)
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
            ids_image = source_ids.keys()
            data_image = self.read({'image': ids_image})['image']
            n2i = name2idx['image']
            for id, data in zip(ids_image, data_image):
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
        """
        arr_dict {'image': dim x A x N arr}
        """

        ctl = calculation_table_list or self.calculation_table_list
        if 'image' in ctl:
            ids_image = ids['image']
            coords = arr_dict['image']
            M = coords.shape[-1]
            for m in range(M):
                coord = coords[..., m]
                label = None
                status = status
                start_time = time.time()
                potential = None
                potentials = None
                gradient = None
                finish_time = None
                positions = None
                forces = None
                datum = (coord, label, status, start_time, potential,
                         potentials, gradient, finish_time, positions, forces)
                id = ids_image[m]
                self.update({'image': [id]}, {'image': [datum]})

    def You_and_I_have_unfinished_business(self, ids,
                                           calculation_table_list=None):
        """
        Kill Bill
        Check calculation of given ids are finished
        """
        ctl = calculation_table_list or self.calculation_table_list
        name2idx = self.name2idx
        intrim_report = []
        if 'image' in ctl:
            n2i = name2idx['image']
            image_data = self.read({'image': ids['image']})['image']
            statuses = np.array([datum[n2i['status']] for datum in image_data])
            if np.all(statuses == 'Finished'):
                intrim_report.append(False)
            else:
                return True

        if 'sbdesc' in ctl:
            n2i = name2idx['sbdesc']
            # source_ids = [id for id in ids['sbdesc']]
            data = self.read({'sbdesc': ids['sbdesc']})['sbdesc']
            ids_image = []
            for datum in data:
                source_id = datum[n2i['source_id']]
                if source_id in ids_image:
                    continue
                ids_image.append(source_id)
            n2i = name2idx['image']
            image_data = self.read({'image': ids_image})['image']
            statuses = np.array([datum[n2i['status']] for datum in image_data])
            if np.all(statuses == 'Finished'):
                intrim_report.append(False)
            else:
                return True
        return np.any(intrim_report)

    def czvf(self, data, tables=None, xzvf=False, name2idx=None):
        """
        data : dict(list of tuples)
        return : data_dict(list of tuples)
        """
        tables = tables or self.tables
        data_dict = OrderedDict()
        # name2idx = name2idx or self.name2idx
        name2idx = self.get_new_name2idx(tables)
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

    def get_new_name2idx(self, tables=None):
        tables = tables or self.tables
        name2idx = {}
        for table_name, entries in tables.items():
            name2idx[table_name] = OrderedDict()
            for idx, name in enumerate(entries.keys()):
                name2idx[table_name][name] = idx
        return name2idx
