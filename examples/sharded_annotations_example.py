"""
Example: Writing and reading sharded neuroglancer precomputed annotations.

Creates synthetic data, writes both sharded and unsharded versions,
reads them back, and verifies they contain the same annotations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.data_model.annotations_utils import parse_annotations


def main():
    rng = np.random.default_rng(42)
    n_points = 1000

    df = pd.DataFrame(
        {
            "x": rng.random(n_points, dtype=np.float32) * 5000,
            "y": rng.random(n_points, dtype=np.float32) * 5000,
            "z": rng.random(n_points, dtype=np.float32) * 500,
            "intensity": rng.integers(0, 255, n_points).astype(np.uint32),
            "category": pd.Categorical(
                rng.choice(["typeA", "typeB", "typeC"], n_points)
            ),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write unsharded
        unsharded_path = tmpdir / "unsharded"
        from_spatialdata_points_to_precomputed_points(
            points=df,
            precomputed_path=unsharded_path,
            points_name="points",
            limit=200,
            starting_grid_shape=(1, 1, 1),
            sharded=False,
        )

        # Write sharded
        sharded_path = tmpdir / "sharded"
        from_spatialdata_points_to_precomputed_points(
            points=df,
            precomputed_path=sharded_path,
            points_name="points",
            limit=200,
            starting_grid_shape=(1, 1, 1),
            sharded=True,
        )

        # Count files
        unsharded_files = list((unsharded_path / "points").rglob("*"))
        unsharded_files = [f for f in unsharded_files if f.is_file()]
        sharded_files = list((sharded_path / "points").rglob("*"))
        sharded_files = [f for f in sharded_files if f.is_file()]
        shard_files = [f for f in sharded_files if f.name.endswith(".shard")]

        print(f"Annotation count: {n_points}")
        print(f"Unsharded total files: {len(unsharded_files)}")
        print(f"Sharded total files: {len(sharded_files)}")
        print(f"  of which .shard files: {len(shard_files)}")

        # Read back both
        df_unsharded = parse_annotations(data_path=unsharded_path)
        df_sharded = parse_annotations(data_path=sharded_path)

        # Compare
        sort_cols = ["intensity", "x", "y", "z"]
        drop_cols = ["__spatial_index__", "__chunk_key__"]
        df_u = (
            df_unsharded.sort_values(by=sort_cols)
            .drop(drop_cols, axis=1)
            .reset_index(drop=True)
        )
        df_s = (
            df_sharded.sort_values(by=sort_cols)
            .drop(drop_cols, axis=1)
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(df_u, df_s, check_names=False)
        print("Sharded and unsharded annotations are identical!")


if __name__ == "__main__":
    main()
