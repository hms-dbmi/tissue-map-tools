import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from cloudvolume import CloudVolume

from tissue_map_tools.data_model.mesh import LegacyMeshInfo
from tissue_map_tools.mesh_writers import (
    from_obj_to_precomputed_meshes,
    write_precomputed_legacy_meshes,
)

# a unit square made of one quad, in two groups together with a triangle
OBJ_WITH_QUAD_AND_GROUPS = """\
o square
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
o triangle
v 0 0 1
v 1 0 1
v 0 1 1
f 5 6 7
"""


def _read_with_cloudvolume(mesh_dir: Path, segment_id: int):
    """Read the meshes back with CloudVolume, which requires a parent volume info."""
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        resolution=[1, 1, 1],
        voxel_offset=[0, 0, 0],
        volume_size=[1, 1, 1],
        mesh=mesh_dir.name,
    )
    vol = CloudVolume(f"file://{mesh_dir.parent}", info=info, progress=False)
    vol.commit_info()
    return vol.mesh.get(segment_id, remove_duplicate_vertices=False)


def test_write_and_read_back_with_cloudvolume(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=50.0)
    box = trimesh.creation.box(extents=[10, 20, 30])
    mesh_dir = tmp_path / "meshes"
    write_precomputed_legacy_meshes(
        {3: (sphere.vertices, sphere.faces), 10: (box.vertices, box.faces)},
        mesh_dir,
    )

    info = LegacyMeshInfo(**json.loads((mesh_dir / "info").read_text()))
    assert info.type == "neuroglancer_legacy_mesh"
    assert info.segment_properties is None

    for segment_id, expected in [(3, sphere), (10, box)]:
        mesh = _read_with_cloudvolume(mesh_dir, segment_id)
        np.testing.assert_allclose(mesh.vertices, expected.vertices, rtol=1e-6)
        np.testing.assert_array_equal(mesh.faces, expected.faces)


def test_from_obj_one_segment_per_file(tmp_path):
    obj_path = tmp_path / "my_cell.obj"
    obj_path.write_text(OBJ_WITH_QUAD_AND_GROUPS)
    sphere_path = tmp_path / "nucleus.obj"
    sphere_path.write_text(
        trimesh.exchange.obj.export_obj(trimesh.creation.icosphere(subdivisions=1))
    )
    mesh_dir = tmp_path / "meshes"
    from_obj_to_precomputed_meshes([obj_path, sphere_path], mesh_dir)

    # sequence input: ids start from 1; the two groups are merged, the quad is split
    mesh = _read_with_cloudvolume(mesh_dir, 1)
    assert len(mesh.vertices) == 7
    assert len(mesh.faces) == 3
    assert (mesh_dir / "2:0").exists()

    # default segment names are the file stems
    info = json.loads((mesh_dir / "info").read_text())
    assert info["segment_properties"] == "segment_properties"
    properties = json.loads((mesh_dir / "segment_properties" / "info").read_text())
    assert properties["@type"] == "neuroglancer_segment_properties"
    assert properties["inline"]["ids"] == ["1", "2"]
    assert properties["inline"]["properties"][0]["values"] == ["my_cell", "nucleus"]


def test_from_obj_with_ids_names_and_transform(tmp_path):
    obj_path = tmp_path / "cell.obj"
    obj_path.write_text(OBJ_WITH_QUAD_AND_GROUPS)
    mesh_dir = tmp_path / "meshes"
    transform = np.diag([1000.0, 1000.0, 1000.0, 1.0])
    transform[:3, 3] = [5, 6, 7]
    from_obj_to_precomputed_meshes(
        {42: obj_path},
        mesh_dir,
        segment_names={42: "custom"},
        transform=transform,
    )

    mesh = _read_with_cloudvolume(mesh_dir, 42)
    assert np.allclose(mesh.vertices.min(axis=0), [5, 6, 7])
    assert np.allclose(mesh.vertices.max(axis=0), [1005, 1006, 1007])
    properties = json.loads((mesh_dir / "segment_properties" / "info").read_text())
    assert properties["inline"]["properties"][0]["values"] == ["custom"]


def test_overwrite(tmp_path):
    box = trimesh.creation.box()
    meshes = {1: (box.vertices, box.faces)}
    mesh_dir = tmp_path / "meshes"
    write_precomputed_legacy_meshes(meshes, mesh_dir)
    with pytest.raises(FileExistsError):
        write_precomputed_legacy_meshes(meshes, mesh_dir)
    write_precomputed_legacy_meshes({2: meshes[1]}, mesh_dir, overwrite=True)
    assert not (mesh_dir / "1:0").exists()
    assert (mesh_dir / "2:0").exists()


@pytest.mark.parametrize(
    "meshes, segment_names",
    [
        ({}, None),
        ({-1: (np.zeros((3, 3)), [[0, 1, 2]])}, None),
        ({1: (np.zeros((3, 3)), [[0, 1, 3]])}, None),
        ({1: (np.zeros((4, 3)), [[0, 1, 2, 3]])}, None),
        ({1: (np.zeros((3, 3)), [[0, 1, 2]])}, {2: "missing"}),
    ],
)
def test_invalid_input(tmp_path, meshes, segment_names):
    mesh_dir = tmp_path / "meshes"
    with pytest.raises(ValueError):
        write_precomputed_legacy_meshes(meshes, mesh_dir, segment_names=segment_names)
    assert not (mesh_dir / "info").exists()
