"""
Tests for shard_util.get_ids_from_mesh_files on sharded meshes.

Regression: shard paths were passed to ShardReader.list_labels relative to the
current working directory, while ShardReader resolves them against its cloudpath
(the dataset root). With a relative data_path this joined the root twice and
raised a misleading `EmptyFileException: ... was zero bytes`.
"""

import json
from pathlib import Path

import pytest
from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from tissue_map_tools.shard_util import get_ids_from_mesh_files

SHARDING = {
    "@type": "neuroglancer_uint64_sharded_v1",
    "preshift_bits": 0,
    "hash": "identity",
    "minishard_bits": 0,
    "shard_bits": 0,
    "minishard_index_encoding": "raw",
    "data_encoding": "raw",
}


def _rel_root(tmp_path: Path) -> str:
    """Unique relative dataset path per test. CloudVolume/CloudFiles cache path
    resolution per path string, so reusing e.g. "out/ds" across tests that chdir into
    different tmp dirs would silently point at an earlier test's dataset."""
    return f"out/{tmp_path.name}"


def make_sharded_mesh_root(root: Path, mesh_ids: list[int], mesh_subdir: str = "mesh") -> Path:
    """Minimal precomputed segmentation root with a *sharded* mesh directory."""
    root.mkdir(parents=True, exist_ok=True)
    cv = CloudVolume(
        cloudpath=f"file://{root.resolve()}",
        info=CloudVolume.create_new_info(
            num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
            resolution=[1, 1, 1], voxel_offset=[0, 0, 0],
            chunk_size=[64, 64, 64], volume_size=[64, 64, 64], mesh=mesh_subdir,
        ),
    )
    cv.commit_info()

    mesh_dir = root / mesh_subdir
    mesh_dir.mkdir()
    (mesh_dir / "info").write_text(json.dumps(
        {"@type": "neuroglancer_legacy_mesh", "sharding": SHARDING}
    ))
    spec = ShardingSpecification.from_dict(dict(SHARDING))
    shards = spec.synthesize_shards({i: b"\x00" * 8 for i in mesh_ids})
    for filename, content in shards.items():
        (mesh_dir / filename).write_bytes(content)
    return root


@pytest.mark.parametrize("path_style", ["relative", "absolute", "dot"])
def test_get_ids_from_sharded_mesh_files_path_styles(tmp_path, monkeypatch, path_style):
    rel = _rel_root(tmp_path)
    make_sharded_mesh_root(tmp_path / rel, mesh_ids=[1, 2, 5])
    monkeypatch.chdir(tmp_path)

    if path_style == "relative":            # the case that raised EmptyFileException
        root = Path(rel)
    elif path_style == "absolute":
        root = (tmp_path / rel).resolve()
    else:                                    # "./"-prefixed relative path
        root = Path(f"./{rel}")

    ids = get_ids_from_mesh_files(data_path=root / "mesh", root_data_path=root)
    assert sorted(ids) == [1, 2, 5]


def test_get_ids_from_sharded_mesh_files_explicit_shard_filename(tmp_path, monkeypatch):
    rel = _rel_root(tmp_path)
    make_sharded_mesh_root(tmp_path / rel, mesh_ids=[3, 4])
    monkeypatch.chdir(tmp_path)
    root = Path(rel)
    shard_name = next((root / "mesh").glob("*.shard")).name

    ids = get_ids_from_mesh_files(
        data_path=root / "mesh", root_data_path=root, shard_filename=shard_name
    )
    assert sorted(ids) == [3, 4]


def test_get_ids_from_sharded_mesh_files_nested_mesh_subdir(tmp_path, monkeypatch):
    # igneous-style mesh dir name, e.g. "mesh_mip_0_err_40.0"
    rel = _rel_root(tmp_path)
    make_sharded_mesh_root(tmp_path / rel, mesh_ids=[1], mesh_subdir="mesh_mip_0_err_40.0")
    monkeypatch.chdir(tmp_path)
    root = Path(rel)

    ids = get_ids_from_mesh_files(data_path=root / "mesh_mip_0_err_40.0", root_data_path=root)
    assert ids == [1]


def test_get_ids_from_mesh_files_accepts_str_data_path(tmp_path, monkeypatch):
    rel = _rel_root(tmp_path)
    make_sharded_mesh_root(tmp_path / rel, mesh_ids=[7])
    monkeypatch.chdir(tmp_path)

    ids = get_ids_from_mesh_files(data_path=f"{rel}/mesh", root_data_path=rel)
    assert ids == [7]