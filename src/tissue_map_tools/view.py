import webbrowser

from typing import Any
from pathlib import Path
import json
import threading
import colorsys
import time
import warnings
import napari
import neuroglancer
from cloudvolume import CloudVolume
from numpy.random import default_rng
import numpy as np
from tissue_map_tools.shard_util import get_ids_from_mesh_files
from tissue_map_tools.utils import is_running_in_notebook, find_free_port
from tissue_map_tools.data_model.annotations import find_annotations_from_cloud_volume
from tissue_map_tools.data_model.annotations_utils import parse_annotations
from vitessce import (
    VitessceConfig,
    CoordinationLevel as CL,
    get_initial_coordination_scope_prefix,
    make_ids_csv_data_url,
    make_colors_csv_data_url,
    ObsSegmentationsNgPrecomputedWrapper,
    ObsPointsNgAnnotationsWrapper,
    CsvWrapper,
)

RNG = default_rng(42)

# `VitessceConfig.web_app()` embeds the *entire* config (including inline `data:` CSV
# URLs for every segment id/color) as URL-encoded JSON in the URL it opens in the
# browser. Past a certain length, opening it just silently fails and the browser lands
# on "about:blank" instead of vitessce.io -- with no exception raised on the Python
# side. This threshold is a heuristic (based on the commonly-cited safe/portable URL
# length across browsers and OS `open`/`webbrowser.open` implementations), not an exact
# limit: URLs somewhat longer than this may still work depending on the browser/OS.
VITESSCE_WEB_APP_URL_WARN_LENGTH = 8_000


def view_precomputed_in_neuroglancer(
    data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_ids: list[int] | None = None,
    show_meshes: bool = True,
    show_annotations: bool = True,
    port: int = 10001,
    viewer: neuroglancer.Viewer | None = None,
    open_browser: bool = True,
    host_local_data: bool = True,
) -> neuroglancer.Viewer:
    if viewer is None:
        viewer = neuroglancer.Viewer()

    cv = CloudVolume(cloudpath=data_path)
    data_type = cv.info["type"]
    # layer_name = layer_name if layer_name is not None else Path(cv.layerpath).name
    layer_name = layer_name if layer_name is not None else cv.info["scales"][0]["key"]

    if viewer is None:
        viewer = neuroglancer.Viewer()
    url = f"precomputed://http://localhost:{port}"
    with viewer.txn() as s:
        if data_type == "image":
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=url,
            )
        elif data_type == "segmentation":
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=url,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if show_meshes:
            if "mesh" in cv.meta.info:
                mesh_subpath = cv.meta.info["mesh"]
                mesh_layer_name = mesh_layer_name if mesh_layer_name else mesh_subpath
                if mesh_ids is None:
                    mesh_ids = get_ids_from_mesh_files(
                        root_data_path=data_path,
                        data_path=Path(data_path) / mesh_subpath,
                    )
                s.layers[mesh_layer_name] = neuroglancer.SegmentationLayer(
                    source=url + f"/{mesh_subpath}",
                    segments=mesh_ids,
                )

        if show_annotations:
            annotations_names = find_annotations_from_cloud_volume(cv)
            for annotation_name in annotations_names:
                s.layers[annotation_name] = neuroglancer.AnnotationLayer(
                    source=url + f"/{annotation_name}",
                )

    if open_browser:
        webbrowser.open(url=viewer.get_viewer_url(), new=2)
    if host_local_data:
        cv.viewer(port=port)

    return viewer


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
    Compute an `initial_camera_state` dict for `view_precomputed_in_vitessce`, centered
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


def view_precomputed_in_vitessce(
    data_path: str,
    show_meshes: bool = True,
    show_annotations: bool = True,
    segments: list[int] | list[str] | None = None,
    segment_colors: dict[str, str] | None = None,
    obs_type_segmentation: str = "cell",
    obs_type_annotation: str = "cell",
    annotation_feature_type: str = "gene",
    obsColorEncoding: str = "obsColors",
    initial_camera_state: dict | None = None,
    camera_presets: list[dict] | None = None,
    show_axis_lines: bool | None = None,
    annotation_options: dict | None = None,
    schema_version: str = "1.0.17",
    name: str = "Precomputed data",
    host_local_data: bool = True,
    use_web_app: bool | None = None,
):
    """
    Build a Vitessce config for a precomputed dataset (segmentation + meshes,
    plus any point annotations found alongside it), and return it as a widget.

    Parameters
    ----------
    data_path
        Path to the root of the precomputed data (local directory or cloud path
        supported by CloudVolume).
    show_meshes
        Whether to add the segmentation + meshes as an `obsSegmentations.ng-precomputed`
        file, wired up via `segmentationLayer`/`segmentationChannel` coordination.
    show_annotations
        Whether to look for and add any point annotation layers found in the
        precomputed data as `obsPoints.ng-annotations` files.
    segments
        Specific segment IDs to select and color. If `None`, all real object IDs
        are auto-discovered from the mesh shard files. Passed via the file's
        `options.segments`.
    segment_colors
        Optional dict mapping segment ID (as a string) to a hex color, e.g.
        `{"612": "#ccbb44"}`. Passed via `options.segmentColors`. If not
        provided, Neuroglancer assigns default colors automatically.
    obs_type_segmentation
        The `obsType` used in coordination for both the segmentation channel.
        Defaults to `"cell"`.
    obs_type_annotation
        The `obsType` used in coordination by the annotation files. Defaults to `"cell"`.
    obsColorEncoding
        How to color the observations
    initial_camera_state
        Optional dict with `'position'`, `'projectionScale'`, and
        `'projectionOrientation'` keys. Passed to the Neuroglancer view via
        `set_props(initialNgCameraState=...)`. If not provided, the camera
        defaults to whatever Neuroglancer/Vitessce chooses on its own, which
        may not point at any real data. See `compute_initial_camera_state` for
        computing one from the dataset's actual content instead of guessing.
    camera_presets
        Optional list of camera preset dicts (`spatialZoom`, `spatialTargetX`,
        `spatialTargetY`, `spatialRotationX`, `spatialRotationOrbit`) passed to the
        `layerControllerBeta` view via `set_props(cameraPresets=...)`.
    show_axis_lines
        Optional bool passed to the Neuroglancer view via
        `set_props(showAxisLines=...)`.
    annotation_options
        Optional dict of extra `options` merged into every `obsPoints.ng-annotations`
        file added when `show_annotations=True`, e.g.
        `{"featureIndexProp": "phenotype", "quantitativeColorProp": "mx1spots",
        "quantitativeColorMax": 58, "projectionAnnotationSpacing": 1}`.
    schema_version
        Vitessce config schema version.
    name
        Name of the Vitessce config.
    host_local_data
        Whether to actually start serving the data locally via
        `CloudVolume.viewer`. Set to `False` if the data is already being served
        elsewhere (e.g. a remote bucket) and `data_path` already points to a
        reachable URL.
    use_web_app
        If None (default), auto-detected: True when running outside a
        Jupyter notebook (plain script or terminal), False when running
        inside one. Set explicitly to override this — e.g. force True in a
        notebook if you specifically want the `vitessce.io` browser tab
        instead of the inline widget.
    Returns
    -------
    A Vitessce widget (via `VitessceConfig.widget()`) if `use_web_app=False`, or
    the `VitessceConfig` object itself if `use_web_app=True` (after opening a
    browser tab and blocking until the user presses Enter).
    """

    cv = CloudVolume(cloudpath=data_path)

    vc = VitessceConfig(schema_version=schema_version, name=name)
    dataset = vc.add_dataset(name)

    # -------------------------------------------------------------------
    # Segmentation + meshes: file `options` (segments/segmentColors) +
    # `coordination_values` (fileUid)
    # -------------------------------------------------------------------
    if show_meshes:
        resolved_ids = segments
        if resolved_ids is None:
            mesh_subpath = cv.meta.info.get("mesh")
            if mesh_subpath is not None:
                resolved_ids = get_ids_from_mesh_files(
                    root_data_path=data_path,
                    data_path=Path(data_path) / mesh_subpath,
                )
        resolved_ids = [str(i) for i in (resolved_ids or []) if str(i) != "0"]
        dataset.add_object(
            ObsSegmentationsNgPrecomputedWrapper(
                data_path=data_path if host_local_data else None,
                data_url=None if host_local_data else data_path,
                coordination_values={"fileUid": "segmentation"},
            )
        )
        # Vitessce adds segments to Neuroglancer either via an obsSets csv file or obsFeatureMatrix.csv / obsColors.csv
        if resolved_ids:
            # Generate a default color per segment if the caller didn't
            # provide one, so segments are always selectable/visible even
            # without explicit colors.
            if segment_colors is None:
                segment_colors = {
                    seg_id: "#{:02x}{:02x}{:02x}".format(
                        *[
                            int(c * 255)
                            for c in colorsys.hsv_to_rgb(
                                i / len(resolved_ids), 0.65, 0.9
                            )
                        ]
                    )
                    for i, seg_id in enumerate(resolved_ids)
                }

            else:
                segment_colors = {str(k): v for k, v in segment_colors.items()}

            dataset.add_object(
                CsvWrapper(
                    csv_url=make_ids_csv_data_url(resolved_ids, use_web_app),
                    data_type="obsFeatureMatrix",
                    coordination_values={
                        "obsType": obs_type_segmentation,
                        "featureType": "feature",
                        "featureValueType": "value",
                    },
                )
            )

            dataset.add_object(
                CsvWrapper(
                    csv_url=make_colors_csv_data_url(
                        {i: segment_colors.get(i, "#ffffff") for i in resolved_ids},
                        use_web_app,
                    ),
                    data_type="obsColors",
                    options={"obsIndex": "id", "obsColors": "color"},
                    coordination_values={"obsType": obs_type_segmentation},
                )
            )

    # -------------------------------------------------------------------
    # Point annotations: file `options` (feature/color props) +
    # `coordination_values` (fileUid, obsType)
    # -------------------------------------------------------------------
    if show_annotations:
        annotation_file_uids = []
        for annotation_name in find_annotations_from_cloud_volume(cv):
            file_uid = f"annotation_{annotation_name}"
            annotation_file_uids.append(file_uid)
            annotation_path = (
                f"{data_path}/{annotation_name}" if host_local_data else None
            )
            annotation_url = (
                None if host_local_data else f"{data_path}/{annotation_name}"
            )
            dataset.add_object(
                ObsPointsNgAnnotationsWrapper(
                    data_path=annotation_path,
                    data_url=annotation_url,
                    coordination_values={
                        "fileUid": file_uid,
                        # TODO: add variants here, centroids vs. molecules
                        "obsType": obs_type_annotation,
                        "featureType": annotation_feature_type,
                    },
                    options=annotation_options,
                )
            )

    # -------------------------------------------------------------------
    # Views
    # -------------------------------------------------------------------
    ng_view = vc.add_view("neuroglancer", dataset=dataset)
    lc_view = vc.add_view("layerControllerBeta", dataset=dataset)
    vc.layout(ng_view | lc_view)

    # -------------------------------------------------------------------
    # View-level props: set_props()
    # -------------------------------------------------------------------
    ng_props: dict[str, Any] = {}
    if initial_camera_state is not None:
        ng_props["initialNgCameraState"] = initial_camera_state
    if show_axis_lines is not None:
        ng_props["showAxisLines"] = show_axis_lines
    if ng_props:
        ng_view.set_props(**ng_props)

    if camera_presets is not None:
        lc_view.set_props(cameraPresets=camera_presets)

    vc.link_views_by_dict(
        [ng_view, lc_view],
        {
            "spatialRenderingMode": "3D",
            "spatialZoom": 0,
            "spatialTargetT": 0,
            "spatialTargetX": 0,
            "spatialTargetY": 0,
            "spatialTargetZ": 0,
            "spatialRotationX": 0,
            "spatialRotationY": 0,
            "spatialRotationOrbit": 0,
        },
        meta=False,
    )

    if show_meshes:
        segmentation_channel = {
            "obsType": obs_type_segmentation,
            "spatialChannelVisible": True,
        }
        if resolved_ids and segment_colors:
            segmentation_channel.update(
                {
                    "featureType": "feature",
                    "featureValueType": "value",
                    "obsColorEncoding": obsColorEncoding,
                }
            )

        vc.link_views_by_dict(
            [ng_view, lc_view],
            {
                "segmentationLayer": CL(
                    [
                        {
                            "fileUid": "segmentation",
                            "spatialLayerOpacity": 1,
                            "spatialLayerVisible": True,
                            "segmentationChannel": CL([segmentation_channel]),
                        }
                    ]
                ),
            },
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"),
        )

    if annotation_file_uids:
        vc.link_views_by_dict(
            [ng_view, lc_view],
            {
                "pointLayer": CL(
                    [
                        {
                            "fileUid": file_uid,
                            "obsType": obs_type_annotation,
                            "spatialLayerOpacity": 1,
                            "spatialLayerVisible": True,
                            "spatialPointStrokeWidth": 0.2,
                            "obsColorEncoding": "geneSelection",
                            "featureValueColormap": "plasma",
                            "spatialLayerLabel": annotation_name,
                            "featureFilterMode": "featureSelection",
                        }
                        for file_uid, annotation_name in zip(
                            annotation_file_uids, find_annotations_from_cloud_volume(cv)
                        )
                    ]
                ),
            },
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsPoints"),
        )

    # -------------------------------------------------------------------
    # Serve locally + return
    # -------------------------------------------------------------------
    if host_local_data:
        cv_port = find_free_port()
        server_thread = threading.Thread(
            target=cv.viewer, kwargs={"port": cv_port}, daemon=True
        )
        server_thread.start()
        time.sleep(1)

    if use_web_app is None:
        use_web_app = not is_running_in_notebook()
    if use_web_app:
        web_app_port = find_free_port()
        # build the URL ourselves first (open=False) so we can check its length before
        # handing it to the browser -- see `VITESSCE_WEB_APP_URL_WARN_LENGTH`.
        vitessce_url = vc.web_app(port=web_app_port, open=False)
        if len(vitessce_url) > VITESSCE_WEB_APP_URL_WARN_LENGTH:
            warnings.warn(
                f"The generated vitessce.io URL is {len(vitessce_url)} characters long, "
                f"which exceeds the {VITESSCE_WEB_APP_URL_WARN_LENGTH}-character heuristic "
                "threshold used here. This usually happens with a large number of "
                "`segments` (explicit or auto-discovered from mesh files), since every "
                "segment id/color is embedded inline in the URL. The browser may silently "
                'fail to open it and land on a blank tab ("about:blank") instead of '
                "vitessce.io, with no error raised here. To fix this, either pass a "
                "smaller explicit `segments` list, or use `use_web_app=False` (the "
                "notebook-inline widget), which serves data over local HTTP instead of "
                "embedding it in the URL and does not have this limitation.",
                stacklevel=2,
            )
        webbrowser.open(vitessce_url)
        input("Server running -- press Enter to stop...\n")
        return vc
    return vc.widget()


def view_precomputed_in_napari(
    data_path: str,
    layer_name: str | None = None,
    mesh_layer_name: str | None = None,
    mesh_ids: list[int] | None = None,
    show_raster: bool = False,
    show_meshes: bool = True,
    show_points: bool = False,
    show_axes: bool = True,
    viewer: napari.Viewer | None = None,
    open: bool = True,
):
    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)

    cv = CloudVolume(data_path)
    layer_name = layer_name if layer_name is not None else cv.info["scales"][0]["key"]

    if show_raster:
        raster = cv.to_dask()
        if raster.ndim != 4:
            raise ValueError(
                f"Expected raster to have 4 dimensions (z, y, x, c), got {raster.ndim}."
            )
        # convert from zyxc to czyx
        raster = raster.transpose(3, 0, 1, 2)
        type = cv.meta.info["type"]
        zyx_scale_factors = cv.meta.info["scales"][0]["resolution"]
        affine = np.diag(zyx_scale_factors + [1])
        if type == "image":
            viewer.add_image(
                raster,
                name=layer_name,
                colormap="gray",
                affine=affine,
            )
        elif type == "segmentation":
            viewer.add_labels(
                raster,
                name=layer_name,
                affine=affine,
            )
        else:
            raise ValueError(f"Unsupported data type: {type}")

    if show_meshes:
        mesh_layer_name = mesh_layer_name if mesh_layer_name else cv.info["mesh"]
        if mesh_ids is None:
            mesh_ids = get_ids_from_mesh_files(
                root_data_path=data_path, data_path=Path(data_path) / mesh_layer_name
            )

        meshes = cv.mesh.get(segids=mesh_ids[1:])

        random_colors = RNG.random((len(mesh_ids) + 1, 3))

        data_mins_xyz: list[float] = []
        data_maxs_xyz: list[float] = []
        for mesh_id in mesh_ids:
            if mesh_id == 0:
                continue
            mesh = meshes[mesh_id]
            if len(mesh) == 0:
                warnings.warn(
                    f"Mesh with ID {mesh_id} is empty. Skipping this mesh.",
                    stacklevel=2,
                )
                continue
            vertices = mesh.vertices
            faces = mesh.faces
            vertex_colors = np.full((len(vertices), 3), random_colors[mesh_id])
            values = np.full(len(vertices), mesh_id)
            surface = (vertices, faces, values)
            viewer.add_surface(
                surface,
                vertex_colors=vertex_colors,
                name=f"{mesh_layer_name}_{mesh_id}",
            )

            if show_axes:
                mins, maxs = np.min(vertices, axis=0), np.max(vertices, axis=0)
                if not data_mins_xyz:
                    data_mins_xyz = mins.tolist()
                    data_maxs_xyz = maxs.tolist()
                else:
                    data_mins_xyz = np.minimum(data_mins_xyz, mins).tolist()
                    data_maxs_xyz = np.maximum(data_maxs_xyz, maxs).tolist()

    if not show_meshes and show_axes:
        warnings.warn(
            "Currently show_axes is only supported when show_meshes is True. Setting "
            "show_axes to False."
        )
        show_axes = False

    if show_points:
        df = parse_annotations(data_path=Path(data_path))
        viewer.add_points(
            df[["x", "y", "z"]].to_numpy(),
            properties={
                col: df[col].to_numpy()
                for col in df.columns
                if col not in ["x", "y", "z"]
            },
            name="points",
            size=1000,
            face_color="white",
        )

    if show_axes and data_maxs_xyz:
        if not show_meshes:
            warnings.warn(
                "Currently show_axes is only supported when show_meshes is True."
            )
        viewer.add_vectors(
            [
                # z axis
                [
                    [0, 0, 0],
                    [0, 0, data_maxs_xyz[2] - data_mins_xyz[2]],
                ],
                # y axis
                [
                    [0, 0, 0],
                    [0, data_maxs_xyz[1] - data_mins_xyz[1], 0],
                ],
                # x axis
                [
                    [0, 0, 0],
                    [data_maxs_xyz[0] - data_mins_xyz[0], 0, 0],
                ],
            ],
            # vectors,
            edge_color=["blue", "green", "red"],
            edge_width=5000,
            name="xyz axes (rgb)",
        )

    if open:
        napari.run()


if __name__ == "__main__":
    # fmt: off
    unique_labels = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24,
    ]
    # fmt: on
    # viewer = view_precomputed_in_neuroglancer(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     # data_path="../../out/20_1_gloms_precomputed_multiscale",
    #     mesh_layer_name="mesh_mip_0_err_40",
    #     mesh_ids=unique_labels,
    # )
    # viewer = view_precomputed_in_napari(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     mesh_layer_name="glom",
    #     # show_raster=True,
    #     show_meshes=True,
    #     mesh_ids=unique_labels,
    # )
    # viewer = view_precomputed_in_napari(
    #     data_path="../../out/20_1_gloms_precomputed",
    #     mesh_layer_name="glom",
    #     mesh_ids=unique_labels,
    # )
    # unique_labels = np.arange(5929).astype(int).tolist()
    # viewer = view_precomputed_in_neuroglancer(
    #     data_path="/Users/macbook/Desktop/moffitt_precomputed",
    #     # mesh_ids=unique_labels,
    # )

    # unique_labels = np.arange(100).astype(int).tolist()
    # viewer = view_precomputed_in_napari(
    #     data_path="/Users/macbook/Desktop/moffitt_precomputed",
    #     mesh_ids=unique_labels,
    # )
    # viewer = view_precomputed_in_neuroglancer(
    #     data_path="/Users/macbook/Desktop/moffitt_precomputed",
    #     # mesh_ids=unique_labels,
    # )
