
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
    def __init__(self, paths_ids=None, table_name='paths', **kwargs):
        self.paths_ids = paths_ids or list()
        self.table_name = table_name

        super().__init__(**kwargs)

    def read(self, ids, prj=None, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read(ids,
                            table_name=table_name, entries=entries, **kwargs)
        return data


    def read_all(self, prj=None, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read_all(table_name=table_name, entries=entries,
                                **kwargs)
        return data

    def write(self, data, prj=None, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        return super().write(data, table_name=table_name,
                             entries=entries, **kwargs)

    def update(self, ids, data, prj=None, table_name=None, entries=None,
               **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().update(ids, data, table_name=table_name, entries=entries,
                       **kwargs)

    def delete(self, ids, prj=None, table_name=None, entries=None, **kwargs):

        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().delete(ids, table_name=table_name, entries=entries, **kwargs)

    def get_paths_data(self):
        None
