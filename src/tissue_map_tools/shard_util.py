from packaging.version import Version
import cloudvolume
from cloudvolume.datasource.precomputed.sharding import (
    ShardReader,
    ShardingSpecification,
)
from cloudvolume import CloudVolume
from pathlib import Path


def get_ids_from_mesh_files(
    data_path: str | Path,
    root_data_path: str | Path,
    shard_filename: str | None = None,
) -> list[int]:
    """
    Get mesh IDs from shard files in a specified directory.

    Parameters
    ----------
    data_path
        path to the folder containing the shard files
    root_data_path
        path to the CloudVolume data, used to access metadata and cache (for
        instance, for meshes this is the parent folder of the mesh directory files).
    shard_filename
        filenames for the shard files (including the .shard extension). If unspecified,
        all shard files in the directory will be considered.
    Returns
    -------
    list[int]
        list of mesh IDs found in the shard files
    """
    is_remote = str(root_data_path).startswith(("http://", "https://", "gs://", "s3://"))
    cloudpath = str(root_data_path) if is_remote else str(Path(root_data_path))
    cv = CloudVolume(cloudpath=cloudpath)
    meta = cv.mesh.meta
    cache = cv.mesh.cache
    if "sharding" in cv.mesh.meta.info:
        # to avoid changing the cv in place
        data = dict(cv.mesh.meta.info["sharding"])
        data["type"] = data["@type"]
        del data["@type"]
        sharding_specification = ShardingSpecification(**data)

        if Version(cloudvolume.__version__) >= Version("12.9.0"):
            shard_reader = ShardReader(
                cloudpath=cloudpath, cache=cache, spec=sharding_specification
            )
        else:
            shard_reader = ShardReader(
                meta=meta, cache=cache, spec=sharding_specification
            )
        # list all shard files
        data_path = Path(data_path)
        if shard_filename is None:
            shard_files = sorted(f.name for f in data_path.glob("*.shard"))
        else:
            shard_files = [shard_filename]
        # ShardReader resolves filenames against its cloudpath (the dataset root), so
        # pass "<mesh subdir>/<file>.shard" -- relative to root_data_path, not to the cwd.
        mesh_subdir = (
            data_path.resolve().relative_to(Path(root_data_path).resolve()).as_posix()
        )
        mesh_ids = []
        for shard_file in shard_files:
            ids =  shard_reader.list_labels(f"{mesh_subdir}/{shard_file}")
            ids = [int(id) for id in ids]
            mesh_ids.extend(ids)
    else:
        index_files = [f.stem for f in Path(data_path).glob("*.index")]
        mesh_ids = [int(f) for f in index_files]
    return mesh_ids
