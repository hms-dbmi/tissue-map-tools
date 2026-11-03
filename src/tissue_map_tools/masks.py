"""
Helpers to inspect and read label masks stored as TIFF files.

The example notebooks read OME-TIFF files, using the OME-XML metadata to find the axis
order. Plain (non-OME) TIFF files have no OME-XML, so that code fails for them. The
functions here detect which kind of file is given and read it accordingly:

- `detect_mask_format()`: "ome-tiff", "tiff" or "ome-zarr".
- `read_tiff_mask()`: lazily read an OME-TIFF or a plain TIFF as a dask array with
  named axes (a subset of `z`, `y`, `x`) and, when the file stores them, the physical
  voxel sizes.
- `tiff_mask_to_labels_element()`: wrap the result as a SpatialData `Labels3DModel`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import dask.array as da
import xmltodict

MaskFormat = Literal["ome-tiff", "tiff", "ome-zarr"]

_TIFF_SUFFIXES = (".tif", ".tiff")
# default chunks (z, y, x) used after reading, a plain TIFF stack is otherwise read one
# plane per chunk, which gives very thin chunks
DEFAULT_CHUNKS_ZYX = (64, 256, 256)
_UM_PER_UNIT = {
    "m": 1e6,
    "mm": 1e3,
    "µm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
}


@dataclass
class TiffMask:
    """A TIFF mask read lazily, with named axes."""

    data: da.Array
    dims: tuple[str, ...]
    # physical voxel size per axis in micrometers, only for the axes that are known
    voxel_size_um: dict[str, float] = field(default_factory=dict)
    format: MaskFormat = "tiff"


def detect_mask_format(path: str | Path) -> MaskFormat:
    """
    Detect the kind of mask file.

    The detection uses the content of the file, not only its name: a TIFF is reported as
    "ome-tiff" when it contains OME-XML metadata (even if the file name ends in `.tif`),
    and as "tiff" otherwise. A directory that contains OME-Zarr `multiscales` metadata
    is reported as "ome-zarr".

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the format is not recognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if path.is_dir():
        if _is_ome_zarr(path):
            return "ome-zarr"
        raise ValueError(
            f"{path} is a directory but does not look like an OME-Zarr store "
            "(no `multiscales` metadata found)."
        )
    if path.suffix.lower() in _TIFF_SUFFIXES:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            return "ome-tiff" if tif.ome_metadata is not None else "tiff"
    raise ValueError(
        f"Unrecognized mask file {path}: expected a .tif/.tiff file or an OME-Zarr "
        "directory."
    )


def _is_ome_zarr(path: Path) -> bool:
    import json

    for name in (".zattrs", "zarr.json"):
        f = path / name
        if f.is_file():
            try:
                attrs = json.loads(f.read_text())
            except ValueError:
                continue
            # zarr v2: top level; zarr v3: nested under "attributes" (and "ome")
            candidates = [attrs, attrs.get("attributes", {})]
            candidates.append(candidates[-1].get("ome", {}))
            if any("multiscales" in c for c in candidates if isinstance(c, dict)):
                return True
    return False


def read_tiff_mask(
    path: str | Path,
    dims: tuple[str, ...] | None = None,
    voxel_size_um: dict[str, float] | None = None,
    chunks_zyx: tuple[int, int, int] | None = DEFAULT_CHUNKS_ZYX,
) -> TiffMask:
    """
    Lazily read an OME-TIFF or a plain TIFF label mask.

    The axis order comes from the file when it is stored there (OME-TIFF and ImageJ
    TIFFs); this avoids guessing it by matching array sizes. A plain TIFF stack only
    has an unnamed page axis (`Q` in tifffile); it is assumed to be `z` with a warning,
    unless `dims` is given.

    Parameters
    ----------
    path
        Path to the `.tif`/`.tiff` file.
    dims
        Names of the array axes, e.g. `("z", "y", "x")`, overriding what is found in the
        file. Needed when the file does not store the axis order and the first axis is
        not `z`.
    voxel_size_um
        Voxel size per axis in micrometers, e.g. `{"x": 0.5, "y": 0.5, "z": 2.0}`;
        overrides the values stored in the file.
    chunks_zyx
        Chunk sizes (z, y, x) applied after reading. A plain TIFF stack is otherwise
        read one plane per chunk; very thin chunks can make the sharded raster pyramid
        fail (see `_regularize_chunks_for_sharded_pyramid` in `converters.py`). Use
        `None` to keep the chunks of the file.

    Returns
    -------
    A `TiffMask` whose `dims` are a subset of `("z", "y", "x")`.
    """
    import tifffile

    path = Path(path)
    if detect_mask_format(path) == "ome-zarr":
        raise ValueError(f"{path} is an OME-Zarr store, not a TIFF file.")

    with tifffile.TiffFile(path) as tif:
        is_ome = tif.ome_metadata is not None
        series = tif.series[0]
        file_axes = series.axes
        file_voxel_size = _voxel_size_um_from_tiff(tif)
        data = da.from_zarr(series.aszarr())
    fmt: MaskFormat = "ome-tiff" if is_ome else "tiff"

    if dims is not None:
        dims = tuple(d.lower() for d in dims)
        if len(dims) != data.ndim:
            raise ValueError(
                f"dims {dims} has {len(dims)} entries but the data has {data.ndim} axes "
                f"(shape {data.shape})."
            )
    else:
        dims = tuple(file_axes.lower())
        if "q" in dims:
            if dims.count("q") > 1:
                raise ValueError(
                    f"The file has several unnamed axes ({file_axes}); please pass dims."
                )
            warnings.warn(
                f"{path.name} does not store its axis order (axes '{file_axes}'); "
                "assuming the stack axis is 'z'. Pass dims=... to override.",
                stacklevel=2,
            )
            dims = tuple("z" if d == "q" else d for d in dims)

    # remove singleton axes such as channel or time
    keep = [i for i, (d, n) in enumerate(zip(dims, data.shape)) if not (n == 1 and d not in "zyx")]
    if len(keep) != data.ndim:
        data = data.reshape([data.shape[i] for i in keep])
        dims = tuple(dims[i] for i in keep)
    unknown = [d for d in dims if d not in ("z", "y", "x")]
    if unknown:
        raise ValueError(
            f"Axes {unknown} (of {dims}, shape {data.shape}) are not z, y or x and are "
            "not singleton; a 3D (or 2D) label mask is required."
        )
    if len(set(dims)) != len(dims):
        raise ValueError(f"Repeated axis names in {dims}.")

    if chunks_zyx is not None:
        size = dict(zip("zyx", chunks_zyx))
        data = data.rechunk(tuple(min(size[d], n) for d, n in zip(dims, data.shape)))

    voxel = {k: v for k, v in file_voxel_size.items() if k in dims}
    if voxel_size_um is not None:
        voxel.update({k.lower(): float(v) for k, v in voxel_size_um.items()})
    return TiffMask(data=data, dims=dims, voxel_size_um=voxel, format=fmt)


def _voxel_size_um_from_tiff(tif) -> dict[str, float]:
    """Physical voxel size per axis in micrometers, from OME-XML or ImageJ metadata."""
    sizes: dict[str, float] = {}
    if tif.ome_metadata is not None:
        pixels = xmltodict.parse(tif.ome_metadata)["OME"]["Image"]
        if isinstance(pixels, list):
            pixels = pixels[0]
        pixels = pixels["Pixels"]
        for ax in "xyz":
            value = pixels.get(f"@PhysicalSize{ax.upper()}")
            if value is not None:
                unit = pixels.get(f"@PhysicalSize{ax.upper()}Unit", "µm")
                sizes[ax] = float(value) * _UM_PER_UNIT.get(unit, 1.0)
    elif tif.is_imagej and tif.imagej_metadata:
        meta = tif.imagej_metadata
        unit = str(meta.get("unit", "um")).replace("micron", "um")
        factor = _UM_PER_UNIT.get(unit)
        if factor is not None and "spacing" in meta:
            sizes["z"] = float(meta["spacing"]) * factor
    return sizes


def tiff_mask_to_labels_element(mask: TiffMask):
    """
    Wrap a `TiffMask` as a SpatialData `Labels3DModel` element.

    If the physical voxel sizes (in micrometers) are known for every axis, a `Scale`
    transformation is attached; otherwise the element has the default identity
    transformation (1 voxel = 1 unit), as in the example notebooks.
    """
    from spatialdata.models import Labels3DModel
    from spatialdata.transformations import Scale

    if len(mask.dims) != 3:
        raise ValueError(
            f"A 3D mask is required to build a Labels3DModel, got dims {mask.dims} "
            f"(shape {mask.data.shape})."
        )
    transformations = None
    if set(mask.dims) <= set(mask.voxel_size_um):
        transformations = {
            "global": Scale([mask.voxel_size_um[d] for d in mask.dims], axes=mask.dims)
        }
    return Labels3DModel.parse(
        mask.data, dims=mask.dims, transformations=transformations
    )
