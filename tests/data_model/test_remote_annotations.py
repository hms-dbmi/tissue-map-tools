"""
Tests for annotation discovery on local and remote (http) precomputed datasets:

- find_annotations_from_cloud_volume: prefers the root info's "annotations" key
  (str or list), falls back to listing local subdirectories, and returns []
  instead of raising for remote datasets without the key.
- compute_initial_camera_state: reads annotation info files through CloudFiles,
  so it also works when data_path is an http(s) URL.
"""

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from cloudvolume import CloudVolume

from tissue_map_tools.data_model.annotations import find_annotations_from_cloud_volume
from tissue_map_tools.utils import compute_initial_camera_state

ANNOTATION_TYPE = "neuroglancer_annotations_v1"


def _make_dataset(
    root: Path,
    annotations: dict[str, tuple[list[float], list[float]]],
    declared: str | list[str] | None,
    extra_dirs: dict[str, dict] | None = None,
) -> Path:
    """
    Minimal precomputed dataset: an empty volume root (so CloudVolume can open it),
    annotation subdirectories with an info file each, optional non-annotation
    subdirectories, and an optional "annotations" key in the root info.
    """
    CloudVolume.from_numpy(
        np.zeros((4, 4, 4), np.uint32),
        vol_path=f"file://{root.resolve()}",
        layer_type="segmentation",
        progress=False,
    )
    for name, (lower, upper) in annotations.items():
        (root / name).mkdir()
        (root / name / "info").write_text(json.dumps(
            {"@type": ANNOTATION_TYPE, "lower_bound": lower, "upper_bound": upper}
        ))
    for name, info in (extra_dirs or {}).items():
        (root / name).mkdir()
        (root / name / "info").write_text(json.dumps(info))

    root_info = json.loads((root / "info").read_text())
    if declared is not None:
        root_info["annotations"] = declared
    (root / "info").write_text(json.dumps(root_info))
    return root


@pytest.fixture
def http_root(tmp_path):
    """Serve tmp_path over http on a free port; yields the base URL."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *a, **k: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ---- find_annotations_from_cloud_volume -------------------------------------
def test_find_annotations_declared_str_local(tmp_path):
    root = _make_dataset(tmp_path / "ds", {"points_a": ([0, 0, 0], [1, 1, 1])}, declared="points_a")
    cv = CloudVolume(f"file://{root.resolve()}")
    assert find_annotations_from_cloud_volume(cv) == ["points_a"]


def test_find_annotations_declared_list_local(tmp_path):
    root = _make_dataset(
        tmp_path / "ds",
        {"points_a": ([0, 0, 0], [1, 1, 1]), "points_b": ([0, 0, 0], [2, 2, 2])},
        declared=["points_a", "points_b"],
    )
    cv = CloudVolume(f"file://{root.resolve()}")
    assert find_annotations_from_cloud_volume(cv) == ["points_a", "points_b"]


def test_find_annotations_undeclared_local_falls_back_to_listing(tmp_path):
    root = _make_dataset(
        tmp_path / "ds",
        {"points_a": ([0, 0, 0], [1, 1, 1])},
        declared=None,
        extra_dirs={"not_annotations": {"@type": "something_else"}},
    )
    cv = CloudVolume(f"file://{root.resolve()}")
    assert find_annotations_from_cloud_volume(cv) == ["points_a"]


def test_find_annotations_undeclared_local_skips_invalid_info(tmp_path):
    root = _make_dataset(tmp_path / "ds", {"points_a": ([0, 0, 0], [1, 1, 1])}, declared=None)
    (root / "broken").mkdir()
    (root / "broken" / "info").write_text("{not json")
    cv = CloudVolume(f"file://{root.resolve()}")
    assert find_annotations_from_cloud_volume(cv) == ["points_a"]


def test_find_annotations_declared_remote(tmp_path, http_root):
    _make_dataset(tmp_path / "ds", {"points_a": ([0, 0, 0], [1, 1, 1])}, declared="points_a")
    cv = CloudVolume(f"{http_root}/ds")
    assert find_annotations_from_cloud_volume(cv) == ["points_a"]


def test_find_annotations_undeclared_remote_returns_empty(tmp_path, http_root):
    # http directories can't be listed: no "annotations" key -> [] instead of FileNotFoundError
    _make_dataset(tmp_path / "ds", {"points_a": ([0, 0, 0], [1, 1, 1])}, declared=None)
    cv = CloudVolume(f"{http_root}/ds")
    assert find_annotations_from_cloud_volume(cv) == []


# ---- compute_initial_camera_state ---------------------------------------------
@pytest.mark.parametrize("remote", [False, True], ids=["local", "http"])
def test_camera_state_fits_annotation_bounds(tmp_path, http_root, remote):
    _make_dataset(
        tmp_path / "ds",
        {"points_a": ([0, 0, 0], [100, 200, 50]), "points_b": ([-100, 0, 0], [0, 400, 50])},
        declared=["points_a", "points_b"],
    )
    data_path = f"{http_root}/ds" if remote else str(tmp_path / "ds")

    state = compute_initial_camera_state(data_path=data_path, zoom_multiplier=1.0)

    # union of both annotation bounding boxes: [-100,0,0] -> [100,400,50]
    assert state["position"] == pytest.approx([0.0, 200.0, 25.0])
    assert state["projectionScale"] == pytest.approx(400.0)


def test_camera_state_skips_missing_annotation_info_remote(tmp_path, http_root):
    # declared name whose info file doesn't exist -> skipped, falls back to the volume bounds
    _make_dataset(tmp_path / "ds", {}, declared="missing_points")
    state = compute_initial_camera_state(data_path=f"{http_root}/ds", zoom_multiplier=1.0)
    assert np.isfinite(state["position"]).all()