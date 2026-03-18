import numpy as np
import pandas as pd
import pytest

import tissue_map_tools.data_model.annotations as ann_module
from tests.data_model.test_annotations import (
    example_info,
    example_annotations,
    example_sharding,
    assert_dict_of_ids_positions_properties_equal,
)
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.data_model.annotations import (
    AnnotationInfo,
    write_annotation_id_index,
    read_annotation_id_index,
    write_spatial_index,
    read_spatial_index,
)
from tissue_map_tools.data_model.annotations_utils import parse_annotations
from tissue_map_tools.data_model.shard_utils import (
    tmt_spec_to_cv_spec,
    compute_annotation_shard_params,
    chunk_name_to_morton_code,
    morton_code_to_chunk_name,
    write_shard_files,
    read_shard_data,
)
from tissue_map_tools.data_model.sharded import ShardingSpecification


def test_tmt_spec_to_cv_spec():
    """Round-trip TMT -> CV spec conversion preserves all fields."""
    tmt_spec = ShardingSpecification(
        **{
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "murmurhash3_x86_128",
            "preshift_bits": 9,
            "minishard_bits": 6,
            "shard_bits": 11,
            "minishard_index_encoding": "gzip",
            "data_encoding": "raw",
        }
    )
    cv_spec = tmt_spec_to_cv_spec(tmt_spec)
    assert cv_spec.hash == "murmurhash3_x86_128"
    assert cv_spec.preshift_bits == 9
    assert cv_spec.minishard_bits == 6
    assert cv_spec.shard_bits == 11
    assert cv_spec.minishard_index_encoding == "gzip"
    assert cv_spec.data_encoding == "raw"


@pytest.mark.parametrize(
    "num_keys,expected_shard_bits,expected_minishard_bits",
    [
        # 0 bits means 1 shard, 1 minishard
        (1000, 0, 0),
        # comes from how compute_shard_params_for_hashed() operates in cloud-volume
        (int(1e7), 4, 9),
    ],
)
def test_compute_annotation_shard_params(
    num_keys, expected_shard_bits, expected_minishard_bits
):
    """Returns valid params for various annotation counts."""
    spec = compute_annotation_shard_params(num_keys)
    assert spec.type == "neuroglancer_uint64_sharded_v1"
    assert spec.hash_function == "murmurhash3_x86_128"
    assert spec.shard_bits == expected_shard_bits
    assert spec.minishard_bits == expected_minishard_bits
    assert spec.preshift_bits == 0


@pytest.mark.parametrize(
    "grid_shape",
    [(2, 2, 1), (3, 3, 3), (1, 1, 1), (4, 2, 3), (10, 10, 10)],
)
def test_morton_code_roundtrip(grid_shape):
    """morton_code_to_chunk_name(chunk_name_to_morton_code(n, s), s) == n for several grid shapes."""
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                chunk_name = f"{x}_{y}_{z}"
                code = chunk_name_to_morton_code(chunk_name, grid_shape)
                recovered = morton_code_to_chunk_name(code, grid_shape)
                assert recovered == chunk_name, (
                    f"Failed for {chunk_name} with grid_shape={grid_shape}: "
                    f"code={code}, recovered={recovered}"
                )


def test_write_read_shard_files(tmp_path):
    """Generic shard write/read round-trip with arbitrary binary data."""
    spec = compute_annotation_shard_params(10)
    data = {i: f"hello_{i}".encode() for i in range(10)}
    write_shard_files(tmp_path / "shards", data, spec)

    # Check .shard files exist
    shard_files = list((tmp_path / "shards").glob("*.shard"))
    assert len(shard_files) > 0

    recovered = read_shard_data(tmp_path / "shards", spec)
    assert data == recovered


def test_write_read_annotation_id_index_sharded(tmp_path):
    """by_id sharded write -> read produces identical annotations, .shard files exist."""
    info_dict = example_info()
    info_dict["by_id"]["sharding"] = example_sharding()
    info = AnnotationInfo(**info_dict)

    annotations = {
        0: (
            [10.0, 20.0, 30.0],
            {"color": [255, 0, 0], "confidence": 0.9, "cell_type": 1},
            {"segment": [101, 102]},
        ),
        1: (
            [40.0, 50.0, 60.0],
            {"color": [0, 255, 0], "confidence": 0.8, "cell_type": 2},
            {"segment": [103]},
        ),
        2: (
            [70.0, 80.0, 90.0],
            {"color": [0, 0, 255], "confidence": 0.7, "cell_type": 0},
            {"segment": []},
        ),
    }

    write_annotation_id_index(root_path=tmp_path, annotations=annotations, info=info)

    # Check that .shard files exist
    index_dir = tmp_path / info.by_id.key
    shard_files = list(index_dir.glob("*.shard"))
    assert len(shard_files) > 0

    # Read back
    read_annotations = read_annotation_id_index(info=info, root_path=tmp_path)

    assert len(annotations) == len(read_annotations)
    for ann_id, original_data in annotations.items():
        read_data = read_annotations[ann_id]
        original_pos, original_props, original_rels = original_data
        read_pos, read_props, read_rels = read_data

        assert original_pos == read_pos
        assert original_props["color"] == read_props["color"]
        assert np.isclose(original_props["confidence"], read_props["confidence"])
        assert original_props["cell_type"] == read_props["cell_type"]
        assert original_rels == read_rels


def test_write_read_spatial_index_sharded(tmp_path):
    """Spatial sharded write -> read produces identical chunks."""
    info_dict = example_info()
    # Ensure sharding is set on all spatial levels
    for spatial in info_dict["spatial"]:
        spatial["sharding"] = example_sharding()
    info = AnnotationInfo(**info_dict)

    annotations = example_annotations()

    for spatial_level in info.spatial:
        write_spatial_index(
            info=info,
            root_path=tmp_path,
            spatial_key=spatial_level.key,
            annotations_by_spatial_chunk=annotations[spatial_level.key],
        )

    # Verify .shard files exist
    for spatial_level in info.spatial:
        index_dir = tmp_path / spatial_level.key
        shard_files = list(index_dir.glob("*.shard"))
        assert len(shard_files) > 0

    # Read back and verify
    for spatial_level in info.spatial:
        read_data = read_spatial_index(
            info=info,
            root_path=tmp_path,
            spatial_key=spatial_level.key,
        )
        original_data = annotations[spatial_level.key]
        assert_dict_of_ids_positions_properties_equal(original_data, read_data)


def test_end_to_end_sharded_pipeline(tmp_path):
    """Full converter pipeline: write sharded and unsharded, compare both against
    the original df and against each other."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.random(n, dtype=np.float32) * 1000,
            "y": rng.random(n, dtype=np.float32) * 1000,
            "z": rng.random(n, dtype=np.float32) * 100,
            "int32_val": rng.integers(0, 100, n).astype(np.int32),
            "categorical": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    kwargs = dict(points=df, limit=30, starting_grid_shape=(1, 1, 1))

    ann_module.RNG = np.random.default_rng(0)
    from_spatialdata_points_to_precomputed_points(
        **kwargs,
        precomputed_path=tmp_path / "unsharded",
        points_name="test",
        sharded=False,
    )
    ann_module.RNG = np.random.default_rng(0)
    from_spatialdata_points_to_precomputed_points(
        **kwargs,
        precomputed_path=tmp_path / "sharded",
        points_name="test",
        sharded=True,
    )

    assert len(list((tmp_path / "sharded" / "test").rglob("*.shard"))) > 0

    df_unsharded = parse_annotations(data_path=tmp_path / "unsharded")
    df_sharded = parse_annotations(data_path=tmp_path / "sharded")

    # Same seed → identical spatial structure → identical iteration order.
    pd.testing.assert_frame_equal(df_unsharded, df_sharded)

    # Both match the original df after sorting by annotation ID (spatial-chunk
    # order ≠ insertion order) and dropping provenance metadata columns.
    drop_cols = ["__spatial_index__", "__chunk_key__"]
    # name of index and dtype of index is allowed to be different
    cmp_kwargs = dict(check_names=False, check_index_type=False)
    for df_parsed in (df_unsharded, df_sharded):
        df_parsed = df_parsed.sort_index().drop(drop_cols, axis=1)
        assert len(df_parsed) == n
        pd.testing.assert_frame_equal(df, df_parsed, **cmp_kwargs)


def test_empty_spatial_chunks_sharded(tmp_path):
    """Empty chunks (count=0) round-trip correctly through sharding."""
    info_dict = example_info()
    for spatial in info_dict["spatial"]:
        spatial["sharding"] = example_sharding()
    info = AnnotationInfo(**info_dict)

    # Only use the second spatial level which has an empty chunk
    spatial_key = "spatial1"
    annotations_data = {
        "0_0_0": [],
        "0_1_0": [],
        "1_0_0": [],
        "1_1_0": [],
    }

    write_spatial_index(
        info=info,
        root_path=tmp_path,
        spatial_key=spatial_key,
        annotations_by_spatial_chunk=annotations_data,
    )

    read_data = read_spatial_index(
        info=info,
        root_path=tmp_path,
        spatial_key=spatial_key,
    )

    # All chunks should be present and empty
    for chunk_name in annotations_data:
        assert chunk_name in read_data
        assert read_data[chunk_name] == []
