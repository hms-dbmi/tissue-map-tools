"""
Writers for meshes in the Neuroglancer precomputed format that do not depend on igneous.

Only the legacy single-resolution, unsharded mesh format is supported. See the
specification at: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md

For multi-resolution and/or sharded meshes, see `tissue_map_tools.igneous_converters`.
"""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray

from tissue_map_tools.data_model.mesh import LegacyMeshInfo

SEGMENT_PROPERTIES_DIR = "segment_properties"


def _validate_segment_id(segment_id: int) -> int:
    segment_id = int(segment_id)
    if not 0 <= segment_id < 2**64:
        raise ValueError(f"Segment ids must be uint64, got {segment_id}.")
    return segment_id


def _parse_transform(transform: ArrayLike | None) -> NDArray[np.float64] | None:
    if transform is None:
        return None
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape == (4, 4):
        transform = transform[:3]
    if transform.shape != (3, 4):
        raise ValueError(
            f"The transform must be a 3x4 or 4x4 affine matrix, got shape "
            f"{transform.shape}."
        )
    return transform


def encode_legacy_mesh_fragment(vertices: ArrayLike, faces: ArrayLike) -> bytes:
    """
    Encode a triangle mesh as a legacy single-resolution mesh fragment.

    Format (little-endian): uint32 number of vertices, float32 [N, 3] vertex
    positions, uint32 [M, 3] triangle vertex indices.
    """
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (M, 3) (triangles only), got {faces.shape}."
        )
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Cannot encode an empty mesh.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain vertex indices out of range.")
    return b"".join(
        [
            np.uint32(len(vertices)).astype("<u4").tobytes(),
            np.ascontiguousarray(vertices).tobytes(),
            np.ascontiguousarray(faces, dtype="<u4").tobytes(),
        ]
    )


def write_precomputed_legacy_meshes(
    meshes: Mapping[int, tuple[ArrayLike, ArrayLike]],
    precomputed_path: str | Path,
    segment_names: Mapping[int, str] | None = None,
    transform: ArrayLike | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write triangle meshes in the legacy single-resolution, unsharded precomputed format.

    Parameters
    ----------
    meshes
        Mapping from segment id (uint64) to a tuple `(vertices, faces)`, where
        `vertices` has shape (N, 3) and `faces` has shape (M, 3).
    precomputed_path
        The directory that will contain the meshes (and their `info` file). It can be
        opened in Neuroglancer as a standalone `precomputed://` source, or referenced by
        the `"mesh"` key of a precomputed segmentation volume `info` file.
    segment_names
        Optional mapping from segment id to a label. If given, an inline segment
        properties directory is written and referenced from the mesh `info` file.
    transform
        Optional 3x4 or 4x4 affine matrix applied to the vertices before writing.
        Neuroglancer interprets the vertex positions in nanometers, so for vertices in
        microns use, e.g., `np.diag([1000, 1000, 1000, 1])`.
    overwrite
        Whether to delete `precomputed_path` first if it already exists.
    """
    precomputed_path = Path(precomputed_path)
    transform = _parse_transform(transform)
    meshes = {_validate_segment_id(k): v for k, v in meshes.items()}
    if len(meshes) == 0:
        raise ValueError("No meshes to write.")
    if segment_names is not None:
        segment_names = {_validate_segment_id(k): v for k, v in segment_names.items()}
        missing = set(segment_names) - set(meshes)
        if missing:
            raise ValueError(
                f"segment_names contains ids without a mesh: {sorted(missing)}."
            )

    if precomputed_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{precomputed_path} already exists; use overwrite=True to replace it."
            )
        shutil.rmtree(precomputed_path)
    precomputed_path.mkdir(parents=True)

    # encode everything before writing the info file, so that invalid meshes don't
    # leave a partially written dataset that looks valid
    fragments = {}
    for segment_id, (vertices, faces) in meshes.items():
        vertices = np.asarray(vertices, dtype=np.float64)
        if transform is not None and vertices.ndim == 2 and vertices.shape[1] == 3:
            vertices = vertices @ transform[:, :3].T + transform[:, 3]
        fragments[segment_id] = encode_legacy_mesh_fragment(vertices, faces)

    for segment_id, fragment in fragments.items():
        fragment_name = f"{segment_id}:0:1"
        (precomputed_path / fragment_name).write_bytes(fragment)
        manifest = {"fragments": [fragment_name]}
        (precomputed_path / f"{segment_id}:0").write_text(json.dumps(manifest))

    info = LegacyMeshInfo(type="neuroglancer_legacy_mesh")
    if segment_names is not None:
        ids = sorted(segment_names)
        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(i) for i in ids],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": [segment_names[i] for i in ids],
                    }
                ],
            },
        }
        (precomputed_path / SEGMENT_PROPERTIES_DIR).mkdir()
        (precomputed_path / SEGMENT_PROPERTIES_DIR / "info").write_text(
            json.dumps(segment_properties)
        )
        info.segment_properties = SEGMENT_PROPERTIES_DIR
    (precomputed_path / "info").write_text(
        json.dumps(info.model_dump(by_alias=True, exclude_none=True))
    )


def from_obj_to_precomputed_meshes(
    obj_paths: Mapping[int, str | Path] | Sequence[str | Path],
    precomputed_path: str | Path,
    segment_names: Mapping[int, str] | None = None,
    transform: ArrayLike | None = None,
    overwrite: bool = False,
) -> None:
    """
    Convert Wavefront OBJ files to legacy single-resolution, unsharded precomputed meshes.

    Each OBJ file becomes one segment: all the objects/groups in a file are merged and
    non-triangular faces are triangulated.

    Parameters
    ----------
    obj_paths
        Mapping from segment id (uint64) to OBJ file path, or a sequence of paths, in
        which case the segment ids are 1, 2, ..., N (0 is usually background).
    precomputed_path
        The directory that will contain the meshes (and their `info` file).
    segment_names
        Optional mapping from segment id to a label. By default, the file names without
        extension are used.
    transform
        Optional 3x4 or 4x4 affine matrix applied to the vertices before writing.
        Neuroglancer interprets the vertex positions in nanometers, so for OBJ files in
        microns use, e.g., `np.diag([1000, 1000, 1000, 1])`.
    overwrite
        Whether to delete `precomputed_path` first if it already exists.
    """
    if not isinstance(obj_paths, Mapping):
        obj_paths = {i + 1: p for i, p in enumerate(obj_paths)}
    paths: dict[int, Path] = {k: Path(v) for k, v in obj_paths.items()}

    meshes = {}
    for segment_id, obj_path in paths.items():
        mesh = trimesh.load(obj_path, file_type="obj", force="mesh", process=False)
        if len(mesh.faces) == 0:
            raise ValueError(f"{obj_path} does not contain any face.")
        meshes[segment_id] = (mesh.vertices, mesh.faces)

    if segment_names is None:
        segment_names = {k: v.stem for k, v in paths.items()}

    write_precomputed_legacy_meshes(
        meshes=meshes,
        precomputed_path=precomputed_path,
        segment_names=segment_names,
        transform=transform,
        overwrite=overwrite,
    )
