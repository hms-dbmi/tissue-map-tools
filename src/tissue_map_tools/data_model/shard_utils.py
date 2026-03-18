"""Utilities for reading and writing sharded neuroglancer precomputed annotations."""

from pathlib import Path

from cloudvolume.datasource.precomputed.sharding import (
    ShardingSpecification as CVShardingSpecification,
    synthesize_shard_files,
    compute_shard_params_for_hashed,
    ShardReader,
)
from cloudvolume.datasource.precomputed.common import (
    compressed_morton_code,
    morton_code_to_gridpt,
)

from tissue_map_tools.data_model.sharded import (
    ShardingSpecification as TMTShardingSpecification,
)


def tmt_spec_to_cv_spec(spec: TMTShardingSpecification) -> CVShardingSpecification:
    """Convert a TMT Pydantic ShardingSpecification to a CloudVolume ShardingSpecification."""
    d = spec.model_dump(by_alias=True, exclude_none=True)
    return CVShardingSpecification.from_dict(d)


def compute_annotation_shard_params(num_keys: int) -> TMTShardingSpecification:
    """Compute shard parameters for annotation data and return a TMT ShardingSpecification."""
    if num_keys <= 0:
        raise ValueError(f"num_keys must be positive, got {num_keys}")
    shard_bits, minishard_bits, preshift_bits = compute_shard_params_for_hashed(
        num_keys
    )
    return TMTShardingSpecification(
        **{
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": "murmurhash3_x86_128",
            "preshift_bits": preshift_bits,
            "minishard_bits": minishard_bits,
            "shard_bits": shard_bits,
            "minishard_index_encoding": "gzip",
            "data_encoding": "raw",
        }
    )


def chunk_name_to_morton_code(
    chunk_name: str, grid_shape: list[int] | tuple[int, ...]
) -> int:
    """Convert a chunk name like '1_2_3' to a compressed Morton code."""
    parts = [int(x) for x in chunk_name.split("_")]
    code = compressed_morton_code(parts, grid_shape)
    return int(code)


def morton_code_to_chunk_name(
    code: int, grid_shape: list[int] | tuple[int, ...]
) -> str:
    """Convert a compressed Morton code back to a chunk name like '1_2_3'."""
    gridpt = morton_code_to_gridpt(code, grid_shape)
    return "_".join(str(int(x)) for x in gridpt)


def write_shard_files(
    shard_dir: Path,
    data: dict[int, bytes],
    tmt_spec: TMTShardingSpecification,
) -> None:
    """Write shard files to disk from a {key: binary} dict.

    Uses synthesize_shard_files (plural) instead of synthesize_shard_file. The docs from
    cloud-volume explain that the former is appropriate for meshes and skeletons, but it
    is appropriate also here:
    1. All data for a given index level is passed at once, so each shard file
       produced is complete — no risk of partial shards from split calls.
    2. Files are named by decimal shard number (cloudvolume convention), instead of
       using a zero-padded hex value (neuroglancer specs). For reading,
       read_shard_data uses cloudvolume's ShardReader which follows the same
       convention, so the naming is internally consistent. Also, the cloudvolume shards
       are recognized by neuroglancer, and the decimal naming is already used for meshes
       and volumes.
    """
    cv_spec = tmt_spec_to_cv_spec(tmt_spec)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files = synthesize_shard_files(cv_spec, data)
    for filename, content in shard_files.items():
        with open(shard_dir / filename, "wb") as f:
            f.write(content)


def read_shard_data(
    shard_dir: Path,
    tmt_spec: TMTShardingSpecification,
) -> dict[int, bytes]:
    """Read all shard files from a directory and return {label: bytes}."""
    cv_spec = tmt_spec_to_cv_spec(tmt_spec)
    reader = ShardReader(cloudpath=None, cache=None, spec=cv_spec)
    result: dict[int, bytes] = {}
    for shard_file in sorted(shard_dir.iterdir()):
        if shard_file.name.endswith(".shard"):
            shard_bytes = shard_file.read_bytes()
            entries = reader.disassemble_shard(shard_bytes)
            for label, content in entries.items():
                result[int(label)] = content
    return result
