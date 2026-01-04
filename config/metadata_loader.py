import json
from pathlib import Path


class MetadataLoader:
    """Load schema metadata from config/schema_config.json."""

    def __init__(self, path: str = "config/schema_config.json"):
        self.path = Path(path)
        self._data = self._load()

    def _load(self):
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    @property
    def tables(self):
        return self._data.get("tables", {})

    @property
    def table_names(self):
        return set(self.tables.keys())

    @property
    def relationships(self):
        return self._data.get("relationships", [])

    def all_columns(self):
        cols = set()
        for spec in self.tables.values():
            for col in spec.get("columns", []):
                cols.add(col.get("name"))
        return cols


metadata = MetadataLoader()
