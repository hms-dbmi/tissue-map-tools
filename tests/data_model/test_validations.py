import numpy as np
import pandas as pd
import pytest

from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.utils import open_precomputed_root

def points(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 1000, n).astype(np.float32),
            "y": rng.uniform(0, 1000, n).astype(np.float32),
            "z": rng.uniform(0, 100, n).astype(np.float32),
            "score": rng.random(n).astype(np.float32),
        }
    )

class TestSegmentationDtype:
    def _labels(self, dtype, fill=1):
        import dask.array as da

        arr = np.zeros((8, 8, 8), dtype=dtype)
        arr[2:5, 2:5, 2:5] = fill
        return da.from_array(arr, chunks=4)

    @pytest.mark.parametrize(
        "dtype,expected",
        [("int64", "uint64"), ("int32", "uint32"), ("int16", "uint32"), ("int8", "uint32")],
    )
    def test_signed_labels_are_cast_to_unsigned(self, dtype, expected):
        from tissue_map_tools.converters import (
            ensure_segmentation_dtype_supported_by_neuroglancer as ensure,
        )

        with pytest.warns(UserWarning, match="Casting labels"):
            out = ensure(self._labels(dtype))
        assert out.dtype == np.dtype(expected)
        assert int(out.max().compute()) == 1

    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
    def test_unsigned_labels_unchanged(self, dtype):
        from tissue_map_tools.converters import (
            ensure_segmentation_dtype_supported_by_neuroglancer as ensure,
        )

        labels = self._labels(dtype)
        assert ensure(labels) is labels

    def test_negative_labels_raise(self):
        from tissue_map_tools.converters import (
            ensure_segmentation_dtype_supported_by_neuroglancer as ensure,
        )

        with pytest.raises(ValueError, match="negative"):
            ensure(self._labels("int64", fill=-1))

    def test_float_labels_raise(self):
        from tissue_map_tools.converters import (
            ensure_segmentation_dtype_supported_by_neuroglancer as ensure,
        )

        with pytest.raises(ValueError, match="integer dtype"):
            ensure(self._labels("float32"))

    def test_precomputed_info_dtype_for_int64_input(self, tmp_path):
        """End to end: an int64 label volume ends up as uint64 in the info file."""
        sd = pytest.importorskip("spatialdata")
        import json

        from tissue_map_tools.converters import (
            from_spatialdata_raster_to_precomputed_raster,
        )

        labels = sd.models.Labels3DModel.parse(
            self._labels("int64").rechunk(4), dims=("z", "y", "x")
        )
        with pytest.warns(UserWarning, match="Casting labels"):
            from_spatialdata_raster_to_precomputed_raster(
                labels, f"file://{tmp_path / 'pc'}"
            )
        info = json.loads((tmp_path / "pc" / "info").read_text())
        assert info["type"] == "segmentation"
        assert info["data_type"] == "uint64"


def test_open_precomputed_root_on_annotation_folder(tmp_path):
    from_spatialdata_points_to_precomputed_points(
        points(), tmp_path, points_name="pts"
    )
    open_precomputed_root(str(tmp_path))  # root works
    with pytest.raises(ValueError, match="parent precomputed root"):
        open_precomputed_root(f"file://{tmp_path / 'pts'}")


def testnormalize_parallel():
    pytest.importorskip("igneous")  # optional dependency (GPL-3.0)
    from tissue_map_tools.igneous_converters import normalize_parallel

    assert normalize_parallel(False) == 1
    assert normalize_parallel(True) is True
    assert normalize_parallel(3) == 3
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            normalize_parallel(bad)