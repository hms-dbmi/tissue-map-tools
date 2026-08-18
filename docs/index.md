<p align="center">
  <img src="assets/teaser.png" alt="tissue-map-tools teaser" width="500"/>
</p>

<h1 align="center">tissue-map-tools</h1>

<p align="center">
  <b>A Python toolkit for scalable 3D spatial biology visualization and analysis</b><br/>
  Bridges SpatialData and Neuroglancer Precomputed formats to enable interactive exploration of 3D tissue maps
</p>

---

## Overview

Biological processes and interactions occur in a spatial, three-dimensional context. While spatial biology technologies promise to elucidate tissue topology at fine-grained resolution, data acquisition, analysis, and visualization have historically focused on 2D tissue sections—missing critical information about cellular relationships across the third dimension.

**tissue-map-tools** addresses these challenges by bridging two complementary standards:

- [**SpatialData**](https://spatialdata.scverse.org/) — an annotated data format for spatial biology (scverse ecosystem)
- [**Neuroglancer Precomputed**](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed) — a scalable format for large-scale 3D segmentations and point clouds (connectomics ecosystem)

The result is an integrated approach for representing, analyzing, and sharing complete 3D spatial biology datasets—including raw volumes, segmentation masks, 3D meshes, and molecular point clouds—with interactive visualization via [Vitessce](https://vitessce.io), [Neuroglancer](https://neuroglancer-demo.appspot.com/), and [napari](https://napari.org/).

### Key Features

| Feature                | Description                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| **Fast**               | Built on OME-NGFF and Neuroglancer Precomputed formats with adaptive sharding and chunking |
| **Interactive**        | Browser-based 3D visualization via Vitessce and Neuroglancer; desktop via napari           |
| **Scalable**           | Handles large volumetric datasets through multi-resolution pyramids and sharded storage    |
| **Adaptive**           | Configurable sharding, chunking, spatial indexing, and level-of-detail parameters          |
| **Reproducible**       | Fully deterministic pipeline delivering the same results across runs                       |
| **scverse-integrated** | Native SpatialData support and compatible with the broader scverse ecosystem               |

---

## Installation

```sh
pip install tissue-map-tools
```

> **Note:** The library is under active development; breaking changes may occur.

### Optional: Mesh Generation

Mesh generation from segmentation masks requires the `igneous-pipeline` dependency, which is licensed under GPL-3.0. Install it separately if needed:

```sh
pip install tissue-map-tools igneous-pipeline
```

> If GPL-3.0 is incompatible with your use case, all other functionality (unsharded raster conversion, point annotations, visualization) remains available without it.

### Development Install

```sh
git clone https://github.com/hms-dbmi/tissue-map-tools
cd tissue-map-tools
uv venv
source .venv/bin/activate
uv sync  # installs examples, dev, and test groups
```

---

## Architecture

tissue-map-tools organizes its functionality into three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Formats                        │
│   OME-Zarr v0.4 · SpatialData Zarr · pandas/dask DataFrame │
└───────────────────────────┬─────────────────────────────────┘
                            │  converters.py / igneous_converters.py
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Neuroglancer Precomputed Format               │
│   Raster Volumes · Segmentation Masks · Meshes (Draco)     │
│   Point Annotations · Sharded Indices · Spatial Indices    │
└───────────────────────────┬─────────────────────────────────┘
                            │  view.py
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Visualization                           │
│         Neuroglancer · napari · Vitessce (browser)         │
└─────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module                            | Description                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `converters.py`                   | Core conversion: OME-Zarr/SpatialData → Precomputed volumes and annotations   |
| `igneous_converters.py`           | Mesh generation and multiscale pyramid creation (requires `igneous-pipeline` (GPL-3.0).) |
| `view.py`                         | Visualization via Neuroglancer and napari                                     |
| `data_model/annotations.py`       | Annotation encoding/decoding, spatial indexing, index I/O                     |
| `data_model/annotations_utils.py` | dtype compatibility, DataFrame ↔ annotation property conversions             |
| `data_model/shard_utils.py`       | Shard file I/O (Morton codes, minishard layout)                               |
| `data_model/mesh.py`              | Pydantic model for multi-LOD Draco mesh metadata                              |
| `data_model/sharded.py`           | Pydantic model for Neuroglancer sharding specifications                       |
| `data_model/print.py`             | CloudVolume → xarray DataTree extraction for volumes and meshes               |
| `shard_util.py`                   | Mesh ID extraction from sharded/unsharded storage                             |
| `config.py`                       | Global debug flags                                                            |

---

## Quick Start

### 1. Convert a Segmentation Mask to Precomputed Format with Meshes

```python
import spatialdata as sd
from tissue_map_tools.igneous_converters import (
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
)

# Load your SpatialData object
sdata = sd.read_zarr("my_dataset.zarr")

# Convert segmentation labels → Precomputed volume + Draco meshes
from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster=sdata["cell_labels"],     # SpatialData labels element
    precomputed_path="./out/precomputed",
    multiscale=True,
    sharded_raster=True,
    sharded_mesh=True,
    nlod=2,                          # 2 levels of detail for meshes
)
```

### 2. Convert Molecular Points to Precomputed Annotations

```python
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.data_model.annotations_utils import (
    make_dtypes_compatible_with_precomputed_annotations,
)

# Prepare the points DataFrame (ensure compatible dtypes)
points_df = make_dtypes_compatible_with_precomputed_annotations(
    sdata["molecules"],
    max_categories=500,
)

# Convert to Neuroglancer annotations
from_spatialdata_points_to_precomputed_points(
    points=points_df,
    precomputed_path="./out/precomputed",
    points_name="molecules",
    limit=10000,       # max points per spatial grid cell
    sharded=True,
)
```

### 3. Visualize in Neuroglancer

```python
from tissue_map_tools.view import view_precomputed_in_neuroglancer

viewer = view_precomputed_in_neuroglancer(
    data_path="./out/precomputed",
    # Automatically detects and loads meshes and annotations
)
```

### 4. Visualize in napari

```python
from tissue_map_tools.view import view_precomputed_in_napari

view_precomputed_in_napari(
    data_path="./out/precomputed",
    show_meshes=True,
    show_points=True,
    show_raster=False,
)
```

---

## API Reference

### `tissue_map_tools.converters`

#### `from_ome_zarr_04_raster_to_precomputed_raster`

Converts an OME-Zarr v0.4 file to Neuroglancer Precomputed format.

```python
from tissue_map_tools.converters import from_ome_zarr_04_raster_to_precomputed_raster

from_ome_zarr_04_raster_to_precomputed_raster(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,  # auto-detected if None
    units_factor: int = 1000,       # pixel size → nm (1000 for µm input)
)
```

**Notes:**

- Auto-detects label vs. image type from the channel count and dtype (integral single-channel → labels).
- Squeezes singleton time and channel dimensions automatically.
- Only scale transformations are currently supported (not scale + translation).

---

#### `from_spatialdata_raster_to_precomputed_raster`

Converts a SpatialData image or labels element to Precomputed format.

```python
from tissue_map_tools.converters import from_spatialdata_raster_to_precomputed_raster

from_spatialdata_raster_to_precomputed_raster(
    raster: DataArray | DataTree,   # SpatialData image or labels element
    precomputed_path: str | Path,
    units_factor: int = 1000,
)
```

**Notes:**

- Only diagonal coordinate transformations are currently supported.
- Pixel sizes are extracted from the SpatialData transformation and converted to nanometers.

---

#### `from_spatialdata_points_to_precomputed_points`

Converts a points DataFrame (SpatialData points element or plain pandas/dask) to Neuroglancer precomputed annotations with a multi-level spatial index.

```python
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points

from_spatialdata_points_to_precomputed_points(
    points: DaskDataFrame | pd.DataFrame,  # must have x, y, z columns
    precomputed_path: str | Path,
    points_name: str | None = None,        # subdirectory name, defaults to f"points_{limit}"
    limit: int = 50000,                    # max annotations per spatial grid cell
    starting_grid_shape: tuple | None = None,  # initial grid dimensions, default (1,1,1)
    sharded: bool = False,                 # enable sharded index storage
)
```

**Supported property dtypes:** `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, `category` (enum).

**Notes:**

- All non-spatial columns are automatically converted to annotation properties.
- Categorical columns are encoded as enum properties.
- A multi-level spatial index is computed using an adaptive KD-tree grid.
- Requires a pre-existing Precomputed root (created automatically if absent).

---

### `tissue_map_tools.igneous_converters`

> Requires `igneous-pipeline` (GPL-3.0).

#### `from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes`

Full pipeline: SpatialData labels → Precomputed segmentation volume → multi-LOD Draco meshes.

```python
from tissue_map_tools.igneous_converters import (
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
)

from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster: DataArray | DataTree,
    precomputed_path: str | Path,
    multiscale: bool = True,              # create multiscale pyramid
    sharded_raster: bool = True,          # shard the volume data
    sharded_mesh: bool = True,            # shard the mesh data
    mesh_name: str | None = None,         # mesh subdirectory name (igneous default if None)
    units_factor: int = 1000,             # pixel size → nm
    object_ids: list[int] | None = None,  # mesh IDs to generate (all if None)
    shape: tuple = (448, 448, 448),       # finest LOD chunk shape
    nlod: int = 0,                        # number of additional LOD levels
    min_chunk_size: tuple = (256,256,256),# coarsest LOD chunk shape
    parallel: int | bool = True,          # parallel processing
)
```

**Mesh LOD guidance:**

- `nlod=0` produces a single resolution mesh.
- Increasing `nlod` adds coarser levels that are used at greater viewing distances.
- `shape` controls the spatial extent of the finest LOD; larger values improve rendering performance.
- `min_chunk_size` sets the coarsest resolution floor—if a mesh is smaller than this, no additional LOD levels are created.
- Monitor actual LOD usage via Neuroglancer: _Settings → Render → Resolution (mesh)_.

---

#### `from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes`

Same as above, but accepts an OME-Zarr v0.4 path as input.

```python
from tissue_map_tools.igneous_converters import (
    from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes,
)

from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    ome_zarr_path: str | Path,
    precomputed_path: str | Path,
    is_labels: bool | None = None,
    multiscale: bool = True,
    sharded_raster: bool = True,
    sharded_mesh: bool = True,
    mesh_name: str | None = None,
    units_factor: int = 1000,
    object_ids: list[int] | None = None,
    shape: tuple = (448, 448, 448),
    nlod: int = 0,
    min_chunk_size: tuple = (256, 256, 256),
    parallel: int | bool = True,
)
```

---

#### `from_precomputed_raster_to_precomputed_meshes`

Generate meshes from an existing Precomputed segmentation volume.

```python
from tissue_map_tools.igneous_converters import from_precomputed_raster_to_precomputed_meshes

from_precomputed_raster_to_precomputed_meshes(
    data_path: str,
    mesh_name: str | None = None,
    object_ids: list[int] | None = None,
    shape: tuple = (448, 448, 448),
    nlod: int = 0,
    min_chunk_size: tuple = (256, 256, 256),
    parallel: int | bool = True,
    sharded: bool = True,
)
```

---

#### `from_precomputed_raster_modify_scales_and_sharding`

Add multiscale pyramid levels and/or convert to sharded storage for an existing Precomputed volume.

```python
from tissue_map_tools.igneous_converters import (
    from_precomputed_raster_modify_scales_and_sharding,
)

from_precomputed_raster_modify_scales_and_sharding(
    data_path: str,
    multiscale: bool,   # create downsampled levels
    sharded: bool,      # convert to sharded format
    num_mips: int = 4,  # number of pyramid levels to create
    parallel: int | bool = True,
)
```

---

### `tissue_map_tools.view`

#### `view_precomputed_in_neuroglancer`

Opens a Neuroglancer viewer displaying a Precomputed volume, its meshes, and any associated annotations.

```python
from tissue_map_tools.view import view_precomputed_in_neuroglancer

viewer = view_precomputed_in_neuroglancer(
    data_path: str,                        # path to Precomputed root
    layer_name: str | None = None,         # display name for the volume layer
    mesh_layer_name: str | None = None,    # display name for the mesh layer
    mesh_ids: list[int] | None = None,     # specific mesh IDs to show (auto-detected if None)
    show_meshes: bool = True,
    show_annotations: bool = True,
    port: int = 10001,                     # local HTTP port for data serving
    viewer: neuroglancer.Viewer | None = None,  # reuse existing viewer
    open_browser: bool = True,
    host_local_data: bool = True,
) -> neuroglancer.Viewer
```

**Returns** the `neuroglancer.Viewer` instance (can be reused or shared as a URL).

---

#### `view_precomputed_in_napari`

Opens a napari viewer displaying a Precomputed dataset with optional raster, mesh, and point layers.

```python
from tissue_map_tools.view import view_precomputed_in_napari

view_precomputed_in_napari(
    data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_ids: list[int] | None = None,
    show_raster: bool = False,
    show_meshes: bool = True,
    show_points: bool = False,
    show_axes: bool = True,    # render XYZ axis vectors (RGB = XYZ)
    viewer: napari.Viewer | None = None,
    open: bool = True,         # call napari.run()
)
```

**Notes:**

- `show_axes` renders three vectors colored red (X), green (Y), blue (Z) scaled to the data extent.
- `show_axes` currently requires `show_meshes=True`.
- Point properties are passed as napari `properties` for color-by-feature support.

---

### `tissue_map_tools.data_model.annotations_utils`

#### `make_dtypes_compatible_with_precomputed_annotations`

Converts a DataFrame to use only dtypes supported by the Neuroglancer annotation format.

```python
from tissue_map_tools.data_model.annotations_utils import (
    make_dtypes_compatible_with_precomputed_annotations,
)

compatible_df = make_dtypes_compatible_with_precomputed_annotations(
    df: pd.DataFrame,
    max_categories: int = 1000,    # max unique values for categorical columns
    check_for_overflow: bool = True,
) -> pd.DataFrame
```

**Conversions applied:**

| Input dtype        | Output dtype                                     |
| ------------------ | ------------------------------------------------ |
| `float64`          | `float32`                                        |
| `int64` / `uint64` | `int32` / `uint32`                               |
| `object` (string)  | `category` (if ≤ `max_categories` unique values) |

---

#### `parse_annotations`

Reads all annotations from a Precomputed directory and returns a flat DataFrame.

```python
from tissue_map_tools.data_model.annotations_utils import parse_annotations

df = parse_annotations(data_path: Path) -> pd.DataFrame
# Returns DataFrame with x, y, z columns + all annotation properties
```

---

### `tissue_map_tools.data_model.annotations`

Low-level annotation encoding/decoding and spatial index computation.

#### `compute_spatial_index`

Builds a multi-level spatial grid index from a set of 3D coordinates.

```python
from tissue_map_tools.data_model.annotations import compute_spatial_index

grid = compute_spatial_index(
    xyz: np.ndarray,                          # (N, 3) float array
    kd_tree: KDTree | None = None,            # pre-built KDTree (optional)
    limit: int = 1000,                        # max points per grid cell
    starting_grid_shape: tuple | None = None, # initial grid (default: (1,1,1))
) -> dict[int, GridLevel]
```

**Returns** a dictionary mapping level index → `GridLevel` object. The grid adapts by splitting cells that exceed the `limit`.

---

#### Data Models

**`AnnotationProperty`** — defines a single typed property for annotations:

| Field         | Type                | Description                                                            |
| ------------- | ------------------- | ---------------------------------------------------------------------- |
| `id`          | `str`               | Property identifier (column name)                                      |
| `type`        | `str`               | Data type (`uint8`–`uint32`, `int8`–`int32`, `float32`, `rgb`, `rgba`) |
| `description` | `str \| None`       | Human-readable label                                                   |
| `values`      | `list[int] \| None` | Enum integer codes (for categorical)                                   |
| `labels`      | `list[str] \| None` | Enum string labels (for categorical)                                   |

**`AnnotationInfo`** — full annotation layer metadata (serialized to `info` JSON):

| Field                         | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `dimensions`                  | Axis units (e.g., `{"x": [1.0, "nm"], ...}`)                 |
| `lower_bound` / `upper_bound` | Spatial extent                                               |
| `annotation_type`             | `POINT`, `LINE`, `AXIS_ALIGNED_BOUNDING_BOX`, or `ELLIPSOID` |
| `properties`                  | List of `AnnotationProperty`                                 |
| `relationships`               | List of `AnnotationRelationship` (links to other objects)    |
| `by_id`                       | ID-based index specification                                 |
| `spatial`                     | List of `AnnotationSpatialLevel` (grid-based indices)        |

---

### `tissue_map_tools.shard_util`

#### `get_ids_from_mesh_files`

Extracts mesh segment IDs from a Precomputed mesh directory (handles both sharded and unsharded storage).

```python
from tissue_map_tools.shard_util import get_ids_from_mesh_files

ids = get_ids_from_mesh_files(
    data_path: str | Path,       # path to mesh subdirectory
    root_data_path: str | Path,  # path to Precomputed root
    shard_filename: str | None = None,
) -> list[int]
```

---

## Examples

### MERFISH Mouse Ileum (Full Workflow)

A complete end-to-end example located in `examples/merfish_mouse_ileum/`:

**Step 0 — Raw data → SpatialData** (`0_raw_to_spatialdata.py`)

Parses raw MERFISH data:

- Reads OME-TIFF image stacks (DAPI, membrane staining)
- Parses transcript coordinates from Baysor and Cellpose segmentations
- Constructs pseudo-3D cell geometries from 2.5D layered sections
- Saves a consolidated `SpatialData` `.zarr` store

**Step 1 — SpatialData → Precomputed** (`1_spatialdata_to_precomputed.py`)

```python
import spatialdata as sd
from tissue_map_tools.igneous_converters import (
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
)
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points
from tissue_map_tools.data_model.annotations_utils import (
    make_dtypes_compatible_with_precomputed_annotations,
)

sdata = sd.read_zarr("merfish_mouse_ileum.sdata.zarr")

# Convert segmentation labels + generate meshes
from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
    raster=sdata["membrane_labels"],
    precomputed_path="./out/merfish_precomputed",
)

# Convert molecular transcripts
points_df = make_dtypes_compatible_with_precomputed_annotations(
    sdata["molecule_baysor"].compute(),
    max_categories=250,
)
# Scale to nanometers (data is in µm)
for ax in ["x", "y", "z"]:
    points_df[ax] = points_df[ax] * 1000

from_spatialdata_points_to_precomputed_points(
    points=points_df,
    precomputed_path="./out/merfish_precomputed",
    points_name="molecule_baysor",
    limit=10000,
    sharded=True,
)
```

**Step 2 — Visualize** (`2_view.py`)

```python
from tissue_map_tools.view import view_precomputed_in_neuroglancer

viewer = view_precomputed_in_neuroglancer(
    data_path="./out/merfish_precomputed",
)
```

**Step 3 — Precomputed → SpatialData** (`3_precomputed_to_spatialdata.py`)

Round-trip conversion from Precomputed back to SpatialData for downstream analysis.

---

### OME-TIFF to Sharded Meshes

For CycIF or other OME-TIFF-based datasets (`examples/invasive/`, `examples/melanoma/`):

```python
from tissue_map_tools.igneous_converters import (
    from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes,
)

from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    ome_zarr_path="./my_data/0",   # bioformats2raw output subdirectory
    precomputed_path="./out/precomputed",
    multiscale=True,
    sharded_raster=True,
    sharded_mesh=True,
    nlod=3,
    shape=(448, 448, 448),
)
```

---

### Sharded Annotations Standalone Example

`examples/sharded_annotations_example.py` demonstrates the annotation pipeline independently:

```python
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tissue_map_tools.converters import from_spatialdata_points_to_precomputed_points

rng = default_rng(42)
N = 500_000

df = pd.DataFrame({
    "x": rng.random(N, dtype=np.float32) * 1000,
    "y": rng.random(N, dtype=np.float32) * 1000,
    "z": rng.random(N, dtype=np.float32) * 500,
    "intensity": rng.integers(0, 255, N, dtype=np.uint32),
    "gene": pd.Categorical(rng.choice(["GeneA", "GeneB", "GeneC"], N)),
})

from_spatialdata_points_to_precomputed_points(
    points=df,
    precomputed_path="./out/annotations_example",
    limit=5000,
    sharded=True,
)
```

---

## Integration Guide

### SpatialData

tissue-map-tools consumes SpatialData elements directly:

| SpatialData.model | Conversion Function                                               |
| ------------------- | ----------------------------------------------------------------- |
| `3D image`      | `from_spatialdata_raster_to_precomputed_raster`                   |
| `3D labels`     | `from_spatialdata_raster_to_precomputed_raster` + mesh generation |
| `3D points`       | `from_spatialdata_points_to_precomputed_points`                   |

**Coordinate transformations:** Only diagonal (scale-only) transformations are currently supported. Translations embedded in transformations are not applied to the output volume (tracked as a known limitation).

**Unit handling:** SpatialData typically stores pixel sizes in micrometers. Use `units_factor=1000` (default) to convert to nanometers as required by Neuroglancer.

---

### Neuroglancer

After conversion, data is served locally via CloudVolume's built-in HTTP server:

```python
viewer = view_precomputed_in_neuroglancer(
    data_path="./out/precomputed",
    port=10001,           # HTTP port for data serving
    open_browser=True,    # auto-open in default browser
)
# Share the URL:
print(viewer.get_viewer_url())
```

The viewer automatically:

- Detects the layer type (`image` vs. `segmentation`)
- Finds and loads mesh layers from the `info` file
- Discovers and adds all annotation layers

---

### napari

```python
from tissue_map_tools.view import view_precomputed_in_napari

view_precomputed_in_napari(
    data_path="./out/precomputed",
    show_raster=True,     # load volume as image/labels layer
    show_meshes=True,     # load individual meshes as surface layers
    show_points=True,     # load annotations as points layer
    show_axes=True,       # add XYZ reference axes (red/green/blue)
    mesh_ids=[1, 2, 5],   # specific IDs to load (auto-detected if None)
)
```

---

### Vitessce

Vitessce provides a fully browser-based, configurable multi-panel interface for spatial biology data. tissue-map-tools produces data in formats that Vitessce supports natively (OME-Zarr for images, Neuroglancer Precomputed for segmentations). Vitessce enables combining spatial images, segmentation overlays, cell metadata, dimensionality reduction embeddings, and gene expression heatmaps in a single shareable URL.

Refer to the [Vitessce documentation](https://vitessce.io/docs/) for configuration details.

---

## Supported Data Types

### Input Formats

| Format                        | Description                             |
| ----------------------------- | --------------------------------------- |
| OME-Zarr v0.4                 | Multiscale images/labels via `ome-zarr` |
| SpatialData `.zarr`           | scverse annotated spatial data          |
| OME-TIFF (via bioformats2raw) | Converted to OME-Zarr first             |
| `pandas.DataFrame`            | Points with x, y, z columns             |
| `dask.DataFrame`              | Lazy-loaded points                      |

### Output Formats

| Format                                     | Contents                                   |
| ------------------------------------------ | ------------------------------------------ |
| Neuroglancer Precomputed                   | Image volumes, segmentation masks          |
| Neuroglancer Precomputed                   | Multi-LOD Draco-compressed meshes          |
| Neuroglancer Annotations v1                | Typed point annotations with spatial index |
| Sharded (`neuroglancer_uint64_sharded_v1`) | Efficient storage for large datasets       |

### Annotation Property Types

| Type                        | Description                                                   |
| --------------------------- | ------------------------------------------------------------- |
| `uint8`, `uint16`, `uint32` | Unsigned integers                                             |
| `int8`, `int16`, `int32`    | Signed integers                                               |
| `float32`                   | Single-precision float                                        |
| `rgb`                       | 3-channel color                                               |
| `rgba`                      | 4-channel color with alpha                                    |
| `category`                  | Categorical/enum (stored as integer codes with string labels) |

---

## Configuration

### Debug Flags (`config.py`)

```python
import tissue_map_tools.config as tmt_config

tmt_config.PRINT_DEBUG = True   # verbose stdout output (default: True)
tmt_config.VISUAL_DEBUG = False  # matplotlib visualizations (default: False)
```

### Sharding Parameters

Sharding splits data into fixed-size shard files using a hash-based index. The `compute_annotation_shard_params()` function automatically selects appropriate parameters based on the number of entries.

For manual control, construct a `ShardingSpecification`:

| Parameter                  | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `preshift_bits`            | Right-shift applied to keys before hashing          |
| `minishard_bits`           | Number of minishards per shard = `2^minishard_bits` |
| `shard_bits`               | Number of shards = `2^shard_bits`                   |
| `minishard_index_encoding` | `raw` or `gzip` compression for minishard index     |
| `data_encoding`            | `raw` or `gzip` compression for chunk data          |

See [docs/shard_binary_format.md](shard_binary_format.md) for the complete binary format specification.

### Mesh Generation Parameters

| Parameter        | Default           | Description                                   |
| ---------------- | ----------------- | --------------------------------------------- |
| `shape`          | `(448, 448, 448)` | Chunk dimensions for the finest mesh LOD      |
| `nlod`           | `0`               | Additional LOD levels (0 = single resolution) |
| `min_chunk_size` | `(256, 256, 256)` | Minimum chunk size at coarsest LOD            |
| `parallel`       | `True`            | Enable parallel mesh generation               |
| `sharded`        | `True`            | Store meshes in sharded format                |

---

## Requirements

- Python ≥ 3.11
- Core dependencies: `cloudvolume`, `ome-zarr`, `zarr`, `numpy`, `dask`, `xarray`, `scikit-learn`, `pydantic`
- Visualization: `neuroglancer`, `napari`, `napari-spatialdata`
- Spatial biology: `spatialdata`
- Mesh generation (optional, GPL-3.0): `igneous-pipeline`, `task-queue`

---

## License

tissue-map-tools is licensed under the **BSD 3-Clause License**.

The optional `igneous-pipeline` dependency is licensed under **GPL-3.0**. If you use igneous-based meshing functions (`igneous_converters` module), your combined work must comply with GPL-3.0 terms. The rest of tissue-map-tools can be used independently under BSD 3-Clause.

---

## Contributing & Contact

Feedback and contributions are welcome! Please:

- Open a [GitHub issue](https://github.com/hms-dbmi/tissue-map-tools/issues) for bugs or feature requests
- Join [this dedicated channel](https://scverse.zulipchat.com/#narrow/channel/545206-tissue-map-tools-dev) or join the discussion on [scverse Zulip](https://scverse.zulipchat.com/)

---

## Citation

If you use tissue-map-tools in your research, please cite the associated paper (preprint forthcoming).
