import argparse
import json
from pathlib import Path

import mlflow


def numeric_leaves(obj, prefix=""):
    """Yield (dotted.path, float) for every int/float leaf; strings/bools become params elsewhere."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for x in obj:
            yield from numeric_leaves(x, prefix)
    else:
        leaf = (prefix.rstrip('.') if prefix else "value"), obj
        try:
            yield leaf[0], float(leaf[1])
        except (TypeError, ValueError):
            pass


