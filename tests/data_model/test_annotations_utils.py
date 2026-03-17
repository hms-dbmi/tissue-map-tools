import numpy as np
import pandas as pd
import pytest

from tissue_map_tools.data_model.annotations_utils import (
    make_dtypes_compatible_with_precomputed_annotations,
)


class TestMakeDtypesCompatible:
    """Tests for make_dtypes_compatible_with_precomputed_annotations."""

    def test_already_supported_dtypes_are_preserved(self):
        """Columns with natively supported dtypes (float32, int32, uint8, etc.)
        should pass through unchanged."""
        df = pd.DataFrame(
            {
                "x": np.array([1.0, 2.0], dtype=np.float32),
                "y": np.array([3, 4], dtype=np.int32),
                "z": np.array([5, 6], dtype=np.uint32),
                "a": np.array([7, 8], dtype=np.int16),
                "b": np.array([9, 10], dtype=np.uint16),
                "c": np.array([11, 12], dtype=np.int8),
                "d": np.array([13, 14], dtype=np.uint8),
            }
        )
        result = make_dtypes_compatible_with_precomputed_annotations(df)

        assert result["x"].dtype == np.float32
        assert result["y"].dtype == np.int32
        assert result["z"].dtype == np.uint32
        assert result["a"].dtype == np.int16
        assert result["b"].dtype == np.uint16
        assert result["c"].dtype == np.int8
        assert result["d"].dtype == np.uint8
        assert list(result.columns) == ["x", "y", "z", "a", "b", "c", "d"]

    def test_mixed_supported_and_convertible(self):
        """A mix of natively supported, convertible, and unsupported dtypes."""
        df = pd.DataFrame(
            {
                "f32": np.array([1.0], dtype=np.float32),
                "f64": np.array([2.0], dtype=np.float64),
                "i32": np.array([3], dtype=np.int32),
                "i64": np.array([4], dtype=np.int64),
                "u8": np.array([5], dtype=np.uint8),
                "label": ["hello"],
            }
        )
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["f32"].dtype == np.float32
        assert result["f64"].dtype == np.float32
        assert result["i32"].dtype == np.int32
        assert result["i64"].dtype == np.int32
        assert result["u8"].dtype == np.uint8
        assert result["label"].dtype.name == "category"

    def test_unsupported_dtype_excluded_with_warning(self):
        df = pd.DataFrame(
            {
                "good": np.array([1.0, 2.0], dtype=np.float32),
                "bad": np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
            }
        )
        with pytest.warns(match="not supported or convertible"):
            result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert "good" in result.columns
        assert "bad" not in result.columns

    def test_float64_converted_to_float32(self):
        df = pd.DataFrame({"val": np.array([1.0, 2.0], dtype=np.float64)})
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["val"].dtype == np.float32

    def test_int64_converted_to_int32(self):
        df = pd.DataFrame({"val": np.array([1, 2], dtype=np.int64)})
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["val"].dtype == np.int32

    def test_float64_overflow_raises(self):
        df = pd.DataFrame(
            {"val": pd.array([np.finfo(np.float32).max * 2], dtype=np.float64)}
        )
        with pytest.raises(ValueError, match="outside float32 range"):
            make_dtypes_compatible_with_precomputed_annotations(df)

    def test_int64_overflow_raises(self):
        df = pd.DataFrame(
            {"val": pd.array([np.iinfo(np.int32).max + 1], dtype=np.int64)}
        )
        with pytest.raises(ValueError, match="outside int32 range"):
            make_dtypes_compatible_with_precomputed_annotations(df)

    def test_object_converted_to_category(self):
        df = pd.DataFrame({"label": ["a", "b", "c"]}, dtype="object")
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["label"].dtype.name == "category"

    def test_string_converted_to_category(self):
        df = pd.DataFrame({"label": ["a", "b", "c"]}, dtype="str")
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["label"].dtype.name == "category"

    def test_bool_converted_to_uint8(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["flag"].dtype == np.uint8

    def test_category_preserved(self):
        df = pd.DataFrame({"cat": pd.Categorical(["x", "y", "x"])})
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert result["cat"].dtype.name == "category"

    def test_column_order_preserved(self):
        df = pd.DataFrame(
            {
                "z": np.array([1], dtype=np.float32),
                "a": np.array([2], dtype=np.int32),
                "m": np.array([3], dtype=np.uint8),
                "n": np.array([4], dtype=np.bool),
            }
        )
        result = make_dtypes_compatible_with_precomputed_annotations(df)
        assert list(result.columns) == ["z", "a", "m", "n"]
