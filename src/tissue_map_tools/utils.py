import socket
from cloudvolume import CloudVolume
import numpy as np
from tissue_map_tools.shard_util import get_ids_from_mesh_files
def docstring_parameter(**kw):
    """Decorator to format docstrings with keyword arguments."""

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**kw)
        return func

    return decorator


def is_running_in_notebook() -> bool:
    """Returns True if running inside a Jupyter notebook/lab (or qtconsole),
    False for a plain script or a plain IPython terminal session."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False
    

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0)) 
        return s.getsockname()[1]


def compute_initial_camera_state(
    data_path: str,
    segments: list[int] | list[str] | None = None,
    camera_segments: list[int] | list[str] | None = None,
    zoom_multiplier: float = 0.2,
    projection_orientation: tuple[float, float, float, float] = (
        -0.636204183101654,
        -0.5028395652770996,
        0.5443811416625977,
        0.2145828753709793,
    ),
) -> dict:
    """
    Compute an `initial_camera_state` dict for `vitessce-visualization`, centered
    and zoomed on the data.

    If `segments` (or `camera_segments`, see below) is given, the camera is centered and
    zoomed on the combined bounding box of just those meshes' vertices -- this is what
    you want when only viewing a handful of segments, since fitting the whole dataset
    would make them look tiny.

    Otherwise, the camera is fit to the combined bounding box of *everything actually
    present* in the dataset: every mesh (if a mesh directory exists) and every point
    annotation layer (if any exist), unioned together. A precomputed dataset may have
    only some of these -- meshes without annotations, annotations without meshes, or
    neither -- so each is only included if present. Only if neither meshes nor
    annotations are found does this fall back to the full raster volume's bounding box
    (which is usually much larger than where the actual content sits).

    Note that a single oversized or spatially distant segment in the list can dominate
    the combined bounding box and defeat the zoom (e.g. one large/far-away segment mixed
    in with several small, clustered ones forces the box -- and therefore the zoomed-out
    camera -- to span the distance between them). Use `camera_segments` to frame the
    camera on a tighter, more representative subset while still displaying the full
    `segments` list.

    Parameters
    ----------
    data_path
        Path to the root of the precomputed data (same as passed to
        `view_precomputed_in_vitessce`).
    segments
        Segment IDs to fit the camera to. Should normally match the `segments`
        argument passed to `view_precomputed_in_vitessce`. If `None`, every available
        mesh/annotation/volume source is considered instead (see above). Ignored if
        `camera_segments` is given.
    camera_segments
        Segment IDs to fit the camera to, when you want the camera framing to be based
        on a different (typically smaller/tighter) set than what's actually displayed
        -- e.g. excluding an outlier segment that would otherwise force the view to
        zoom out. Defaults to `segments` when not given, otherwise has precedence over
        `segments`.
    zoom_multiplier
        Multiplier applied to the bounding box's largest dimension. Larger values zoom
        out more
    projection_orientation
        Passed straight through as `projectionOrientation` (a quaternion controlling
        the 3D viewing angle) -- this is a viewing-angle preference, not something
        derived from the data.

    Returns
    -------
    dict
        A dict with `position`, `projectionScale`, and `projectionOrientation` keys,
        ready to pass as `initial_camera_state` to `view_precomputed_in_vitessce`.
    """
    cv = CloudVolume(cloudpath=data_path)

    fit_segments = camera_segments if camera_segments is not None else segments
    if fit_segments:
        segment_ids = [int(segment) for segment in fit_segments]
        meshes = cv.mesh.get(segids=segment_ids)
        vertices = np.concatenate([mesh.vertices for mesh in meshes.values()], axis=0)
        min_corner = vertices.min(axis=0)
        max_corner = vertices.max(axis=0)
    else:
        corners: list[tuple[np.ndarray, np.ndarray]] = []

        mesh_subpath = cv.meta.info.get("mesh")
        if mesh_subpath is not None:
            all_segment_ids = [
                segment_id
                for segment_id in get_ids_from_mesh_files(
                    root_data_path=data_path, data_path=Path(data_path) / mesh_subpath
                )
                if segment_id != 0
            ]
            if all_segment_ids:
                meshes = cv.mesh.get(segids=all_segment_ids)
                vertices = np.concatenate(
                    [mesh.vertices for mesh in meshes.values()], axis=0
                )
                corners.append((vertices.min(axis=0), vertices.max(axis=0)))

        for annotation_name in find_annotations_from_cloud_volume(cv):
            with open(Path(data_path) / annotation_name / "info") as f:
                annotation_info = json.load(f)
            corners.append(
                (
                    np.array(annotation_info["lower_bound"], dtype=float),
                    np.array(annotation_info["upper_bound"], dtype=float),
                )
            )

        if corners:
            min_corner = np.min([corner[0] for corner in corners], axis=0)
            max_corner = np.max([corner[1] for corner in corners], axis=0)
        else:
            resolution = np.array(cv.resolution, dtype=float)
            min_corner = np.array(cv.bounds.minpt, dtype=float) * resolution
            max_corner = np.array(cv.bounds.maxpt, dtype=float) * resolution

    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    projection_scale = float(np.max(size)) * zoom_multiplier

    return {
        "position": center.tolist(),
        "projectionScale": projection_scale,
        "projectionOrientation": list(projection_orientation),
    }

