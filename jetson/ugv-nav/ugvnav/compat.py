"""Every numpy/scipy version difference lives here. Board: scipy 1.3.3 uses
n_jobs; scipy >= 1.6 uses workers."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import scipy
from scipy.spatial import cKDTree


def _ver(mod) -> Tuple[int, ...]:
    out = []
    for p in mod.__version__.split(".")[:2]:
        d = "".join(c for c in p if c.isdigit())
        out.append(int(d) if d else 0)
    return tuple(out)


SCIPY_VERSION = _ver(scipy)
_PARALLEL_KW = "workers" if SCIPY_VERSION >= (1, 6) else "n_jobs"


def kdtree_query(tree: cKDTree, points: np.ndarray, k: int, n_jobs: int = -1):
    try:
        return tree.query(points, k=k, **{_PARALLEL_KW: n_jobs})
    except TypeError:
        return tree.query(points, k=k)


def kdtree_query_radius(tree: cKDTree, points: np.ndarray, radius: float, n_jobs: int = -1):
    try:
        return tree.query_ball_point(points, r=radius, **{_PARALLEL_KW: n_jobs})
    except TypeError:
        return tree.query_ball_point(points, r=radius)


def describe_environment() -> str:
    return "numpy {} | scipy {} | cKDTree parallel keyword: '{}'".format(
        np.__version__, scipy.__version__, _PARALLEL_KW)
