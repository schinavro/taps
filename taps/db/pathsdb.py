
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
    ),
    def __init__(self, entries=None, table_name=None, **kwargs):
        self.entries = entries or self.entries
        self.table_name = table_name or self.table_name

        super().__init__(**kwargs)

    def read(self, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read(table_name=table_name, entries=entries, **kwargs)
        return data

    def read_all(self, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        data = super().read_all(table_name=table_name, entries=entries,
                                **kwargs)
        return data

    def write(self, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name

        super().write(table_name=table_name, entries=entries, **kwargs)

    def update(self, table_name=None, entries=None, **kwargs):
        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().update(table_name=table_name, entries=entries, **kwargs)

    def delete(self):
        prj = prj or self.prj
        entries = entries or self.entries
        table_name = table_name or self.table_name
        super().write(table_name=table_name, entries=entries, **kwargs)
