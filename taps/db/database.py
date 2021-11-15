import os
import time
import pickle
import sqlite3
import numpy as np
from numpy import newaxis as nax
from sqlite3 import OperationalError
from collections import OrderedDict
from scipy.spatial import KDTree

from taps.paths import Paths

from taps.utils.antenna import packing, unpacking, dictify
# from taps.descriptor import SphericalHarmonicDescriptor

class Database:

    metadata=OrderedDict(
        ctimeout='real'
    ),

    def __init__(self, filename=None, entries=None, metadata=None,
                 _cache=None, **kwargs):
        self.entries = entries or self.entries
        self.metadata = metadata or self.metadata
        self.filename = filename or self.__class__.__name__ + '.db'
        self._cache = _cache or dict()

    def create_table(self, filename=None, table_name=None, **kwargs):
        filename = filename or self.filename
        table_name = table_name or self.table_name

        directory, prefix = os.path.split(filename)
        if directory == '':
            directory = '.'
        if not os.path.exists(directory):
            os.makedirs(directory)

        conn = sqlite3.connect(filename)
        c = conn.cursor()

        create_statement = 'CREATE TABLE IF NOT EXISTS ' + table_name
        entry_statement = ' ID INTEGER PRIMARY KEY AUTOINCREMENT, '
        for key, dtype in self.entries.items():
            entry_statement += '%s %s, ' % (key, dtype)
        create_statement += ' ({})'.format(entry_statement[:-2])
        c.execute(create_statement)
        conn.commit()
        conn.close()

    def read(self, ids, filename=None, table_name=None, entries=None,
             **kwargs):
        """
        ids: [1, 2, 3, ..]
        return [{'coord':...}, ]
        """
        filename = filename or self.filename
        entries = entries or self.entries
        table_name = table_name or self.table_name

        if not os.path.exists(filename):
            self.create_table(table_name=table_name)

        conn = sqlite3.connect(filename)
        c = conn.cursor()

        keys = list(entries.keys())
        data = []
        select_statement = "SELECT " + ', '.join(keys) + " FROM "
        select_statement += table_name + " WHERE rowid="
        for id in ids:
            c.execute(select_statement + str(id))
            cache = c.fetchone()
            datum = {}
            for i in range(len(keys)):
                if cache[i] is None:
                    datum[keys[i]] = None
                elif entries[keys[i]] == 'blob':
                    datum[keys[i]] = unpacking(cache[i], includesize=True)[0][0]
                else:
                    datum[keys[i]] = cache[i]
            data.append(datum)
        conn.commit()
        conn.close()
        return data

    def read_all(self, filename=None, table_name=None, entries=None, **kwargs):
        """
        return list of dict
        """

        filename = filename or self.filename
        entries = entries or self.entries
        table_name = table_name or self.table_name

        if not os.path.exists(filename):
            self.create_table(table_name=table_name)

        conn = sqlite3.connect(filename)
        c = conn.cursor()

        keys = list(entries.keys())
        select_statement = "SELECT " + ', '.join(keys) + " FROM "
        select_statement += table_name
        c.execute(select_statement)
        cache = c.fetchall()
        data = []
        for dat in cache:
            cache = dat
            datum = {}
            for i in range(len(keys)):
                if cache[i] is None:
                    datum[keys[i]] = None
                elif entries[keys[i]] == 'blob':
                    datum[keys[i]] = unpacking(cache[i], includesize=True)[0][0]
                else:
                    datum[keys[i]] = cache[i]
            data.append(datum)
        conn.commit()
        conn.close()
        return data

    def write(self, data, filename=None, table_name=None, entries=None,
                 **kwargs):
        """
        Parameters
        ==========

        data : list of dict
            [{'coords':..., 'time': ..., 'lable': ...}, ]
        return : list of int
            [5, 6, 7, ...]
        """
        filename = filename or self.filename
        entries = entries or self.entries
        table_name = table_name or self.table_name

        if not os.path.exists(filename):
            self.create_table(table_name=table_name)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        c.execute('SELECT max(ROWID) from ' + table_name)
        maxid = c.fetchone()[0] or 0
        ids = []
        for datum in data:
            dat = []
            keys = []
            for k, v in datum.items():
                keys.append(k)
                if entries[k] == 'blob':
                    if isinstance(v, np.ndarray):
                        dat.append(packing(v))
                    elif v is None:
                        dat.append(v)
                    else:
                        dat.append(packing(**dictify(v, ignore_cache=False)))
                else:
                    dat.append(v)
            insert_statement = 'INSERT INTO ' + table_name
            insert_statement += ' ({})'.format(', '.join(keys))
            insert_statement += ' VALUES '
            insert_statement += '({})'.format(', '.join(['?'] * len(keys)))
            c.execute(insert_statement, dat)
            id_statement = "SELECT ROWID FROM " + table_name
            id_statement += " WHERE ROWID > %d ORDER BY ROWID ASC" % maxid
            c.execute(id_statement)
            ids.extend(c.fetchall()[0])
            maxid += 1

        conn.commit()
        conn.close()
        return ids

    def update(self, ids, data, filename=None, table_name=None, entries=None,
               **kwargs):
        filename = filename or self.filename
        entries = entries or self.entries
        table_name = table_name or self.table_name
        assert len(ids) == len(data)

        if not os.path.exists(filename):
            self.create_table(table_name=table_name, entries=entries)

        # data = czvf(data, entries=entries)

        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        M = len(ids)
        for i in range(M):
            datum, id = data[i], ids[i]
            update_statement = 'UPDATE ' + table_name + ' SET '
            dat_list_with_id = []
            keys = []
            for k, v in datum.items():
                keys.append(k)
                if entries[k] == 'blob':
                    if isinstance(v, np.ndarray):
                        dat_list_with_id.append(packing(v))
                    elif v is None:
                        dat_list_with_id.append(v)
                    else:
                        pack = packing(**dictify(v, ignore_cache=False))
                        dat_list_with_id.append(pack)
                else:
                    dat_list_with_id.append(v)
            dat_list_with_id.append(id)
            update_statement += '=?, '.join(keys) + '=? '
            update_statement += 'WHERE rowid=?'
            c.execute(update_statement, dat_list_with_id)
        conn.commit()
        conn.close()

    def delete(self, ids, filename=None, table_name=None, entries=None,
               **kwargs):
        # Currently only supports deleting all informations of specific id
        """
        ids : list
            ids = [3, 4, 5]
        """
        filename = filename or self.filename
        entries = entries or self.entries
        table_name = table_name or self.table_name

        if not os.path.exists(filename):
            self.create_table(table_name=table_name, entries=entries)

        conn = sqlite3.connect(filename)
        c = conn.cursor()

        delete_statement = 'DELETE FROM ' + table_name + ' WHERE id in '
        delete_id_list = ', '.join([str(id) for id in ids])
        delete_statement += '({})'.format(delete_id_list)
        c.execute(delete_statement)
        conn.commit()
        conn.close()

    def count(self, query=None, key_mapper=None, table_name='variables'):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        select_statement = "SELECT COUNT(*) FROM %s" % table_name
        if query is not None:
            key_mapper = key_mapper or self.key_mapper
            for mapper in key_mapper.items():
                query = query.replace(*mapper)
            select_statement += " WHERE " + query
        c.execute(select_statement)
        count = c.fetchone()
        conn.commit()
        conn.close()
        return count[0]

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
