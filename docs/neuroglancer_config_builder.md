# Building Neuroglancer Vitessce configs — a guide

This module (`tissue_map_tools/vitessce_configs/`) builds Vitessce configs
for Neuroglancer views from small dataclasses instead of hand-written
`VitessceConfig` calls. It can be helpful for creating configs with more than one segmentation or
annotation layer, or real gene/cell-type data alongside them.

## 1. Figure out what specs you need

| You have... | Use |
|---|---|
| A segmentation + meshes (precomputed) | `SegmentationLayerSpec` |
| Centroids/Points/transcript annotations (precomputed) | `AnnotationLayerSpec` |
| A CSV (obsSets, obsFeatureMatrix, obsEmbedding) | `TabularObsSpec` |
| A `.sdata.zarr` store (gene matrix, cell-type sets) | `SpatialDataObsSpec` |

A vitessce config is just: build the specs for what you have, pass them into
`build_neuroglancer_config`.

## 2. Minimal example

```python
from tissue_map_tools.view import compute_initial_camera_state
from tissue_map_tools.vitessce_configs.layer_specs import SegmentationLayerSpec, AnnotationLayerSpec
from tissue_map_tools.vitessce_configs.neuroglancer_config_builder import build_neuroglancer_config

data_url = "https://your-host/data/my_dataset"  # or data_path="/local/path" instead

vc = build_neuroglancer_config(
    name="My dataset",
    segmentations=[SegmentationLayerSpec(file_uid="cells", data_url=data_url)],
    annotations=[AnnotationLayerSpec(file_uid="points", data_url=f"{data_url}/points", obs_type="point")],
    initial_camera_state=compute_initial_camera_state(data_path=data_url),
)
vc
```

Local data: use `data_path=` instead of `data_url=` on any spec — Vitessce
serves it automatically, no manual server setup needed.

## 3. Common options, by spec
Defined in [`src/tissue_map_tools/vitessce_configs/layer_specs.py`](../src/tissue_map_tools/vitessce_configs/layer_specs.py):
**`SegmentationLayerSpec`**
- `obs_color_encoding`: `"cellSetSelection"` (default, colors by named groups) or `"geneSelection"` (colors by a gene's expression).
- With no `segments` given, IDs are auto-discovered and grouped into one set.
- Pass `obs_sets_csv=TabularObsSpec(...)` to use your own groups instead.
- Set `auto_generate_obs_sets=False` when real sets come from elsewhere (`obs_sets_csv` or a `spatialdata_obs` entry) — otherwise you get a redundant auto-generated set alongside the real ones. It only controls the auto-generated CSV, not mesh-ID discovery — discovery is skipped separately, and automatically, whenever `mesh_load_projection_scale_threshold` is set on the config (on-demand loading — see §4).



**Not needed when `obs_sets_csv` is set** — that already takes priority over auto-generation on its own.

**`AnnotationLayerSpec`**
- `feature_selection`, `feature_value_colormap`, `feature_value_colormap_range` — for gene-driven point coloring.
- `options` — the wrapper's own settings (`projectionAnnotationSpacing`, `transform`, `quantitativeColorProp`, etc.).

**`SpatialDataObsSpec`**
- `obs_feature_matrix_path` must be the **full** path including the table, e.g. `"tables/my_table/X"` (not just `"X"` — see §4).
- If your feature matrix and cell-type sets are in different tables, use two specs sharing the same `sdata_path`/`sdata_url`.

**`build_neuroglancer_config`**
- `extra_view_types=["featureList"]` adds a gene-search sidebar; `"obsSets"` is added automatically whenever needed.
- `mesh_load_projection_scale_threshold` enables on-demand mesh loading (only meshes in view get fetched).


 ## 4. On-demand mesh loading
 
Vitessce supports on-demand mesh loading for performance: cell centroids
are loaded as a lightweight points layer first, and only once you zoom in
past `mesh_load_projection_scale_threshold` do the actual meshes load — and
only for the segments currently in view, not the whole dataset.

This requires a centroids points layer, which can be built from a CSV of
`x`/`y`/`z` ( any property columns) using
`from_spatialdata_points_to_precomputed_points` from
`tissue_map_tools.converters` — it handles the spatial index, sharding, and
annotation properties for you (see its docstring for
`limit`/`starting_grid_shape`/`sharded`).

```python
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points

from_spatialdata_points_to_precomputed_points(
    points=my_points_df,             # x, y, z  any property columns
    precomputed_path="/local/output/points",
)
```

When building the config, **give the segmentation and the annotation layer
the same `obs_type`** — on-demand loading correlates which segments are
currently visible in the viewport with the matching point data by that
shared key, so if they differ (e.g. `"cell"` vs `"point"`), that correlation
breaks.

```python
SegmentationLayerSpec(
    file_uid="cells", 
    data_url=segmentation_url, 
    obs_type="cell",
    auto_generate_obs_sets=False,        # skip full mesh-ID discovery
    obs_color_encoding="geneSelection",
    feature_type="gene", 
    feature_selection=["MY_GENE"],
    feature_value_colormap="plasma", feature_value_colormap_range=(0.0, 1.0),
)
AnnotationLayerSpec(
    file_uid="points", 
    data_path="/local/output/points", 
    obs_type="cell",  # must match segmentation's obs_type
    feature_type="gene",
)
```
```python
build_neuroglancer_config(..., mesh_load_projection_scale_threshold=1200)
```

## 5. Gotchas worth knowing up front

**Gene selection not recoloring points/meshes?** 
- The per-layer channel
  needs its own `"featureType"` key set — not just the file's
  `coordinationValues`. Both `SegmentationLayerSpec`/`AnnotationLayerSpec`
  set this correctly when `obs_color_encoding="geneSelection"`.

**`obs_feature_matrix_path` is not relative to `table_path`** — pass the
  full path.

**`obsColorEncoding: "obsColors"` isn't supported here** — it needs a
  per-segment color CSV this module doesn't generate, and separately hits
  [vitessce-python#517](https://github.com/vitessce/vitessce-python/issues/517)
  (a `vc.widget()`-only crash). Use `"cellSetSelection"` or `"geneSelection"`.

**Re-running a cell and the loading indicator does not go away / kernel feels stuck?** 
- Run `import vitessce; vitessce.data_server.stop_all()` at the top of the cell
  before rebuilding — this stops *every* server from the session, not just
  the current `vc`'s. If the problem persists, please shutdown all kernels for a clean restart.

## References

- [`vitessce-python` data wrappers](https://vitessce.github.io/vitessce-python/api_data.html)
- [`VitessceConfig` API](http://python-docs.vitessce.io/api_config.html)
- `tissue_map_tools.utils` — `compute_initial_camera_state`
