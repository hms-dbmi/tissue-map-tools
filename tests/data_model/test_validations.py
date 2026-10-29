import numpy as np
import pandas as pd
import pytest

from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.utils import open_precomputed_root


def _points(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 1000, n).astype(np.float32),
            "y": rng.uniform(0, 1000, n).astype(np.float32),
            "z": rng.uniform(0, 100, n).astype(np.float32),
            "score": rng.random(n).astype(np.float32),
        }
    )


def test_open_precomputed_root_on_annotation_folder(tmp_path):
    from_spatialdata_points_to_precomputed_points(
        _points(), tmp_path, points_name="pts"
    )
    open_precomputed_root(str(tmp_path))  # root works
    with pytest.raises(ValueError, match="parent precomputed root"):
        open_precomputed_root(f"file://{tmp_path / 'pts'}")


def test_normalize_parallel():
    pytest.importorskip("igneous")  # optional dependency (GPL-3.0)
    from tissue_map_tools.igneous_converters import _normalize_parallel

    assert _normalize_parallel(False) == 1
    assert _normalize_parallel(True) is True
    assert _normalize_parallel(3) == 3
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            _normalize_parallel(bad)