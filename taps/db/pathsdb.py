
from taps.db import Database
from collections import OrderedDict

class PathsDatabase(Database):
    """
    Future plan
      Save : Paths -> dictionary -> json+ binary -> sqlite
      load : Sqlite -> json+ binary -> dictionary -> Paths

    """
    entries=OrderedDict(
        paths='blob',
    )
    def __init__(self, data_ids=None, table_name='paths', **kwargs):
        super().__init__(**kwargs)
        self.data_ids = data_ids or list()
        self.table_name = table_name


    def read(self, ids, table_name=None, entries=None, **kwargs):
        table_name = table_name or self.table_name
        entries = entries or self.entries
        data = super().read(ids,
                            table_name=table_name, entries=entries, **kwargs)
        return data


    def read_all(self, table_name=None, entries=None, **kwargs):
        table_name = table_name or self.table_name
        entries = entries or self.entries
        data = super().read_all(table_name=table_name, entries=entries,
                                **kwargs)
        return data

    def read_latest(self, table_name=None, entries=None, **kwargs):
        table_name = table_name or self.table_name
        entries = entries or self.entries
        data = super().read_latest(table_name=table_name, entries=entries,
                                   **kwargs)
        return data


    def write(self, data, table_name=None, entries=None, **kwargs):
        table_name = table_name or self.table_name
        entries = entries or self.entries
        return super().write(data, table_name=table_name,
                             entries=entries, **kwargs)

    def update(self, ids, data, table_name=None, entries=None,
               **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().update(ids, data, table_name=table_name, entries=entries,
                       **kwargs)

    def delete(self, ids, table_name=None, entries=None, **kwargs):

        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().delete(ids, table_name=table_name, entries=entries, **kwargs)

    def get_paths_data(self, ids=None, table_name=None, entries=None,
                       **kwargs):
        data_ids = data_ids or self.data_ids
        entries = entries or self.entries
        table_name = table_name or self.table_name

        if data_ids is None:
            paths_data = self.read_all()
        else:
            paths_data = self.read(data_ids)

        data = []
        for datum in paths_data:
            data.append(datum['paths'])
        return data
