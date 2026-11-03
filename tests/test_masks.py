import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from tissue_map_tools.masks import detect_mask_format, read_tiff_mask  # noqa: E402


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, (6, 40, 50)).astype("uint16")  # (z, y, x), distinct sizes


@pytest.fixture
def plain_tiff(tmp_path, stack):
    path = tmp_path / "plain.tiff"
    tifffile.imwrite(path, stack)
    return path


@pytest.fixture
def ome_tiff(tmp_path, stack):
    path = tmp_path / "mask.ome.tiff"
    tifffile.imwrite(
        path,
        stack,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 2.0,
        },
    )
    return path


def test_detect_formats(plain_tiff, ome_tiff, tmp_path):
    assert detect_mask_format(plain_tiff) == "tiff"
    assert detect_mask_format(ome_tiff) == "ome-tiff"
    # content decides, not the file name
    renamed = tmp_path / "ome_but_named_tif.tif"
    renamed.write_bytes(ome_tiff.read_bytes())
    assert detect_mask_format(renamed) == "ome-tiff"


def test_detect_ome_zarr(tmp_path):
    zarr_dir = tmp_path / "x.zarr"
    zarr_dir.mkdir()
    (zarr_dir / ".zattrs").write_text('{"multiscales": [{}]}')
    assert detect_mask_format(zarr_dir) == "ome-zarr"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="OME-Zarr"):
        detect_mask_format(other)


def test_detect_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_mask_format(tmp_path / "missing.tif")
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(ValueError, match="Unrecognized"):
        detect_mask_format(txt)


def test_read_ome_tiff(ome_tiff, stack):
    mask = read_tiff_mask(ome_tiff)
    assert mask.format == "ome-tiff"
    assert mask.dims == ("z", "y", "x")
    assert mask.voxel_size_um == {"x": 0.5, "y": 0.5, "z": 2.0}
    np.testing.assert_array_equal(mask.data.compute(), stack)


def test_read_plain_tiff_warns_and_assumes_z(plain_tiff, stack):
    with pytest.warns(UserWarning, match="does not store its axis order"):
        mask = read_tiff_mask(plain_tiff)
    assert mask.format == "tiff"
    assert mask.dims == ("z", "y", "x")
    assert mask.voxel_size_um == {}
    np.testing.assert_array_equal(mask.data.compute(), stack)


def test_read_plain_tiff_with_dims_and_voxel_size(plain_tiff, stack):
    mask = read_tiff_mask(
        plain_tiff, dims=("z", "y", "x"), voxel_size_um={"x": 1, "y": 1, "z": 3}
    )  # no warning when dims are given
    assert mask.voxel_size_um == {"x": 1.0, "y": 1.0, "z": 3.0}


def test_dims_length_mismatch(plain_tiff):
    with pytest.raises(ValueError, match="entries"):
        read_tiff_mask(plain_tiff, dims=("y", "x"))


def test_rechunk_default_and_disable(plain_tiff):
    with pytest.warns(UserWarning):
        assert read_tiff_mask(plain_tiff, chunks_zyx=(2, 16, 16)).data.chunksize == (2, 16, 16)
    with pytest.warns(UserWarning):
        assert read_tiff_mask(plain_tiff, chunks_zyx=None).data.chunksize == (1, 40, 50)


def test_imagej_axes_and_spacing(tmp_path, stack):
    path = tmp_path / "ij.tif"
    tifffile.imwrite(
        path, stack, imagej=True, metadata={"axes": "ZYX", "spacing": 2.0, "unit": "um"}
    )
    mask = read_tiff_mask(path)
    assert mask.dims == ("z", "y", "x")
    assert mask.voxel_size_um == {"z": 2.0}


def test_singleton_channel_axis_removed(tmp_path):
    arr = np.ones((1, 6, 40, 50), dtype="uint8")
    path = tmp_path / "czyx.ome.tiff"
    tifffile.imwrite(path, arr, ome=True, metadata={"axes": "CZYX"})
    mask = read_tiff_mask(path)
    assert mask.dims == ("z", "y", "x")
    assert mask.data.shape == (6, 40, 50)


def test_non_singleton_extra_axis_raises(tmp_path):
    arr = np.ones((2, 6, 40, 50), dtype="uint8")
    path = tmp_path / "czyx2.ome.tiff"
    tifffile.imwrite(path, arr, ome=True, metadata={"axes": "CZYX"})
    with pytest.raises(ValueError, match="not z, y or x"):
        read_tiff_mask(path)