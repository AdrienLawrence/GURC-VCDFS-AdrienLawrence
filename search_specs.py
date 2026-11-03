"""Transitional NB201 specification imports; use search_registry for dispatch."""

import sys

from bench201 import specs

sys.modules[__name__] = specs
