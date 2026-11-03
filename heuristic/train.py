"""Transitional import/CLI shim for bench201.heuristic.train."""

import importlib
import sys

if __name__ == "__main__":
    module = importlib.import_module("bench201.heuristic.train")
    if hasattr(module, "main"):
        module.main()
else:
    sys.modules[__name__] = importlib.import_module("bench201.heuristic.train")
