import yaml

class DotDict(dict):
    """
    A dict that supports attribute-style access and recursively converts
    nested dicts/lists of dicts into DotDicts.
    """
    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(kwargs)
        for k, v in d.items():
            super().__setitem__(k, self._convert(v))

    @staticmethod
    def _convert(o):
        if isinstance(o, dict) and not isinstance(o, DotDict):
            return DotDict(o)
        if isinstance(o, list):
            return [DotDict._convert(v) for v in o]
        return o

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"{type(self).__name__} has no attribute '{key}'") from e

    def __setattr__(self, key, value):
        self[key] = self._convert(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert(value))

    def setdefault(self, key, default=None):
        return super().setdefault(key, self._convert(default))

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = self._convert(v)


def load_config(path: str) -> DotDict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return DotDict(raw)


def set_value_at_path(obj: DotDict, path: str, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, DotDict())
    cur[parts[-1]] = value


def merge_config(config: DotDict, overrides):
    """
    Merge key=value CLI overrides. You can still use this, but the baseline
    behavior expects everything to live in the YAML.
    """
    if not overrides:
        return config
    if isinstance(overrides, str):
        overrides = [overrides]

    import yaml as _yaml
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override: {ov}. Use key=value format.")
        key, val_str = ov.split("=", 1)
        key = key.strip()
        try:
            val = _yaml.safe_load(val_str)
        except _yaml.YAMLError:
            val = val_str
        set_value_at_path(config, key, val)
    return config
