import webbrowser
import warnings
from typing import Any
from pathlib import Path
from cloudvolume import CloudVolume
import csv, io
from urllib.parse import quote

from .layer_specs import SegmentationLayerSpec, AnnotationLayerSpec, TabularObsSpec, SpatialDataObsSpec
from tissue_map_tools.shard_util import get_ids_from_mesh_files
from vitessce import (
    VitessceConfig, CoordinationLevel as CL, CoordinationType as ct,
    hconcat, vconcat, get_initial_coordination_scope_prefix,
    CsvWrapper, ObsSegmentationsNgPrecomputedWrapper, ObsPointsNgAnnotationsWrapper,
    SpatialDataWrapper,
)

from tissue_map_tools.utils import is_running_in_notebook, find_free_port

# `VitessceConfig.web_app()` embeds the *entire* config (including inline `data:` CSV
# URLs for every segment id/color) as URL-encoded JSON in the URL it opens in the
# browser. Past a certain length, opening it just silently fails and the browser lands
# on "about:blank" instead of vitessce.io -- with no exception raised on the Python
# side. This threshold is a heuristic (based on the commonly-cited safe/portable URL
# length across browsers and OS `open`/`webbrowser.open` implementations), not an exact
# limit: URLs somewhat longer than this may still work depending on the browser/OS.
VITESSCE_WEB_APP_URL_WARN_LENGTH = 8_000

def _add_segmentation(dataset, spec: SegmentationLayerSpec, use_web_app: bool, skip_segment_discovery: bool = False):
    resolved_ids = spec.segments
    # Discovery finds every real segment id via CloudVolume + mesh shard files, needed
    # to auto-generate a single-group obsSets CSV. Skipped when the caller already
    # supplied obs_sets_csv (nothing to discover for), or when on-demand mesh loading
    # is active (skip_segment_discovery, driven by the config-wide
    # mesh_load_projection_scale_threshold) — the viewport resolves visible segments
    # dynamically at render time in that case, so no upfront id list is needed.
    if resolved_ids is None and spec.obs_sets_csv is None and not skip_segment_discovery:
        cv_path = spec.data_path or spec.data_url
        if cv_path:
            cv = CloudVolume(cloudpath=cv_path)
            mesh_subpath = cv.meta.info.get("mesh")
            if mesh_subpath is not None:
                resolved_ids = get_ids_from_mesh_files(
                    root_data_path=cv_path, data_path=Path(cv_path) / mesh_subpath,
                )
    resolved_ids = [str(i) for i in (resolved_ids or []) if str(i) != "0"]
    dataset.add_object(ObsSegmentationsNgPrecomputedWrapper(
        data_path=spec.data_path, data_url=spec.data_url,
        coordination_values={"fileUid": spec.file_uid},
        options=spec.options or None,
    ))
    added_obs_sets = False
    if spec.obs_sets_csv is not None:
        _add_tabular(dataset, spec.obs_sets_csv)
        added_obs_sets = True
    elif resolved_ids and spec.auto_generate_obs_sets:
        set_name = spec.obs_set_name or spec.spatial_layer_label or spec.file_uid
        dataset.add_object(CsvWrapper(
            csv_url=make_obs_set_csv_data_url(resolved_ids, set_name, for_web_app=use_web_app),
            data_type="obsSets",
            options={"obsIndex": "id", "obsSets": [{"name": set_name, "column": set_name}]},
            coordination_values={"obsType": spec.obs_type},
        ))
        added_obs_sets = True

    channel = {"obsType": spec.obs_type, "spatialChannelVisible": True}
    if added_obs_sets or spec.obs_color_encoding == "geneSelection":
        channel[ct.OBS_COLOR_ENCODING] = spec.obs_color_encoding
    if spec.spatial_channel_color is not None:
        channel["spatialChannelColor"] = spec.spatial_channel_color
    if spec.obs_color_encoding == "geneSelection":
        if spec.feature_type is not None:
            channel["featureType"] = spec.feature_type
        if spec.feature_value_type is not None:
            channel["featureValueType"] = spec.feature_value_type
        if spec.feature_selection is not None:
            channel[ct.FEATURE_SELECTION] = spec.feature_selection
        if spec.feature_value_colormap is not None:
            channel[ct.FEATURE_VALUE_COLORMAP] = spec.feature_value_colormap
        if spec.feature_value_colormap_range is not None:
            channel[ct.FEATURE_VALUE_COLORMAP_RANGE] = spec.feature_value_colormap_range

    channel_dict = {
        "fileUid": spec.file_uid,
        "spatialLayerOpacity": 1,
        "spatialLayerVisible": True,
        "spatialLayerLabel": spec.spatial_layer_label or spec.file_uid,
        "segmentationChannel": CL([channel]),
    }
    return channel_dict, added_obs_sets

def _add_annotation(dataset, spec: AnnotationLayerSpec):
    dataset.add_object(ObsPointsNgAnnotationsWrapper(
        data_path=spec.data_path,
        data_url=spec.data_url,
        coordination_values={
            "fileUid": spec.file_uid,
            "obsType": spec.obs_type,
            "featureType": spec.feature_type,
        },
        options=spec.options or None,
    ))
    channel = {
        "fileUid": spec.file_uid,
        "obsType": spec.obs_type,
        "spatialLayerOpacity": 1,
        "spatialLayerVisible": True,
        "spatialLayerColor": spec.spatial_layer_color,
        "spatialPointStrokeWidth": spec.spatial_point_stroke_width,
        "spatialLayerLabel": spec.spatial_layer_label or spec.file_uid,
        "featureType": spec.feature_type, 
        ct.OBS_COLOR_ENCODING: spec.obs_color_encoding,
    }
    if spec.feature_value_colormap is not None:
        channel[ct.FEATURE_VALUE_COLORMAP] = spec.feature_value_colormap
    if spec.feature_value_colormap_range is not None:
        channel[ct.FEATURE_VALUE_COLORMAP_RANGE] = spec.feature_value_colormap_range
    if spec.feature_selection is not None:
        channel[ct.FEATURE_SELECTION] = spec.feature_selection
        channel["featureFilterMode"] = "featureSelection"
    return channel


def _add_tabular(dataset, spec: TabularObsSpec):
    dataset.add_object(CsvWrapper(
        csv_path=spec.csv_path,
        csv_url=spec.csv_url,
        data_type=spec.data_type,
        options=spec.options or None, 
        coordination_values=spec.coordination_values,
    ))

def _add_spatialdata(dataset, spec: SpatialDataObsSpec) -> bool:
    coordination_values = {"obsType": spec.obs_type}
    if spec.feature_type is not None:
        coordination_values["featureType"] = spec.feature_type
    dataset.add_object(SpatialDataWrapper(
        sdata_path=spec.sdata_path,
        sdata_url=spec.sdata_url,
        table_path=spec.table_path,
        obs_feature_matrix_path=spec.obs_feature_matrix_path,
        obs_set_paths=spec.obs_set_paths,
        obs_set_names=spec.obs_set_names,
        coordination_values=coordination_values,
    ))
    return spec.obs_set_paths is not None  # whether this contributed real obsSets

def make_obs_set_csv_data_url(
    ids: list[str], set_name: str, set_column: str = "Segment Type", for_web_app: bool = False,
) -> str:
    """
    Build a `data:` URL containing a small inline obsSets.csv with an `id`
    column and a single categorical column (`set_column`), assigning every
    id to the same group label (`set_name`).
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", set_column])
    writer.writerows([[i, set_name] for i in ids])
    encoded = quote(buf.getvalue())
    if for_web_app:
        encoded = quote(encoded, safe="")
    return f"data:text/csv,{encoded}"


def assemble_neuroglancer_config(
    segmentations: list[SegmentationLayerSpec] = (),
    annotations: list[AnnotationLayerSpec] = (),
    tabular_obs: list[TabularObsSpec] = (),
    name: str = "Precomputed data",
    schema_version: str = "1.0.17",
    initial_camera_state: dict | None = None,
    show_axis_lines: bool | None = None,
    mesh_load_projection_scale_threshold: float | None = None,
    layer_per_feature_for_points: bool | None = None,
    extra_view_types: list[str] = (),
    spatial_rendering_mode: str = "3D",
    spatial_zoom: float = 0,
    spatial_target_t: float = 0,
    spatial_target_x: float = 0,
    spatial_target_y: float = 0,
    spatial_target_z: float = 0,
    spatial_rotation_x: float = 0,
    spatial_rotation_y: float = 0,
    spatial_rotation_z: float = 0,
    spatial_rotation_orbit: float = 0,
    use_web_app: bool = False,
    spatialdata_obs: list[SpatialDataObsSpec] = (),
) -> VitessceConfig:
    """
    Build a VitessceConfig from segmentation/annotation/tabular/spatialdata
    layer specs and return it directly -- no widget, no web_app, no serving,
    no blocking. This is the shared core used by both build_neuroglancer_config
    (the notebook/script entry point, which adds widget/web_app/blocking on
    top) and tests, which need the bare config without side effects.

    Parameters
    ----------
    segmentations
        Segmentation + mesh layers to add, each as an `obsSegmentations.ng-precomputed`
        file wired up via `segmentationLayer`/`segmentationChannel` coordination.
        Empty by default.
    annotations
        Point annotation layers to add, each as an `obsPoints.ng-annotations` file
        wired up via `pointLayer` coordination. Empty by default.
    tabular_obs
        CSV obs sources (obsSets, obsEmbedding, obsFeatureMatrix) to add
        alongside the spatial layers, e.g. for a gene expression matrix or
        cell-type sets read from a plain CSV. For the same kind of data
        read from a .sdata.zarr store instead, use spatialdata_obs. Empty
        by default.
    name
        Name of the Vitessce config. Defaults to `"Precomputed data"`.
    schema_version
        Vitessce config schema version. Defaults to `"1.0.17"`.
    initial_camera_state
        Optional dict with `'position'`, `'projectionScale'`, and
        `'projectionOrientation'` keys, passed to the Neuroglancer view via
        `set_props(initialNgCameraState=...)`. See `compute_initial_camera_state`
        for computing one from the dataset's actual content instead of guessing.
    show_axis_lines
        Optional bool passed to the Neuroglancer view via `set_props(showAxisLines=...)`.
    mesh_load_projection_scale_threshold
        Optional float passed to the Neuroglancer view via
        `set_props(meshLoadProjectionScaleThreshold=...)`. Maximum projectionScale
        at which meshes start loading — higher means meshes load at lower zoom levels.
        Setting it disables mesh-ID discovery for every segmentation layer - in support of on-demand mesh loading
    layer_per_feature_for_points
        Optional bool passed to the `layerControllerBeta` view via
        `set_props(layerPerFeatureForPoints=...)`.
    extra_view_types
        Additional view component names to add alongside the Neuroglancer and
        layerControllerBeta views, e.g. `["featureList", "obsSets", "scatterplot"]`.
        Empty by default.
    spatial_rendering_mode, spatial_zoom, spatial_target_t, spatial_target_x,
    spatial_target_y, spatial_target_z, spatial_rotation_x, spatial_rotation_y,
    spatial_rotation_z, spatial_rotation_orbit
        Initial values for the corresponding spatial coordination types, linked
        across the Neuroglancer and layerControllerBeta views. 
    use_web_app
        Whether the generated obsSets CSV `data:` URLs should be encoded for
        embedding in a `vitessce.io` URL (True) or a locally-served widget
        (False, default). This function never opens a browser or returns a
        widget itself — see build_neuroglancer_config for that.
    spatialdata_obs
        SpatialData obs sources (feature matrix, cell sets) read directly from a .sdata.zarr store. 
        Empty by default.

    Returns
    -------
    The assembled `VitessceConfig` object -- no widget, no serving, no
    blocking. Pass it to `build_neuroglancer_config`'s tail (or call
    `.widget()` / `.web_app()`) to actually view it.
    """
    if use_web_app is None:
        use_web_app = not is_running_in_notebook()

    vc = VitessceConfig(schema_version=schema_version, name=name)
    dataset = vc.add_dataset(name=name)

    for t in tabular_obs:
        _add_tabular(dataset, t)
    spatialdata_added_obs_sets = [_add_spatialdata(dataset, s) for s in spatialdata_obs]
    seg_results = [
    _add_segmentation(dataset, s, use_web_app,
                       skip_segment_discovery=mesh_load_projection_scale_threshold is not None)
        for s in segmentations
    ]
    seg_channels = [channel for channel, _ in seg_results]
    any_obs_sets = any(added for _, added in seg_results) or any(spatialdata_added_obs_sets)

    point_layers = [_add_annotation(dataset, a) for a in annotations]

    ng_view = vc.add_view("neuroglancer", dataset=dataset)
    ng_props: dict[str, Any] = {}
    if initial_camera_state is not None:
        ng_props["initialNgCameraState"] = initial_camera_state
    if show_axis_lines is not None:
        ng_props["showAxisLines"] = show_axis_lines
    if mesh_load_projection_scale_threshold is not None:
        ng_props["meshLoadProjectionScaleThreshold"] = mesh_load_projection_scale_threshold
    if ng_props:
        ng_view.set_props(**ng_props)

    lc_view = vc.add_view("layerControllerBeta", dataset=dataset)
    if layer_per_feature_for_points is not None:
        lc_view.set_props(layerPerFeatureForPoints=layer_per_feature_for_points)

    view_types = list(extra_view_types)
    if any_obs_sets and "obsSets" not in view_types:
        view_types = view_types + ["obsSets"]
    extra_views = [vc.add_view(v, dataset=dataset) for v in view_types]

    vc.link_views_by_dict([ng_view, lc_view], {
        "spatialRenderingMode": spatial_rendering_mode,
        "spatialZoom": spatial_zoom,
        "spatialTargetT": spatial_target_t,
        "spatialTargetX": spatial_target_x,
        "spatialTargetY": spatial_target_y,
        "spatialTargetZ": spatial_target_z,
        "spatialRotationX": spatial_rotation_x,
        "spatialRotationY": spatial_rotation_y,
        "spatialRotationZ": spatial_rotation_z,
        "spatialRotationOrbit": spatial_rotation_orbit,
    }, meta=False)

    if seg_channels:
        vc.link_views_by_dict([ng_view, lc_view],
            {"segmentationLayer": CL(seg_channels)},
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"))

    if point_layers:
        vc.link_views_by_dict([ng_view, lc_view],
            {"pointLayer": CL(point_layers)},
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsPoints"))

    vc.layout(hconcat(ng_view, vconcat(lc_view, *extra_views)))

    return vc


def build_neuroglancer_config(*args, use_web_app: bool | None = None, **kwargs):
    """
        All other parameters are forwarded to assemble_neuroglancer_config --
        see its docstring for segmentations/annotations/tabular_obs/spatialdata_obs,
        camera/view options, and initial spatial coordination values.
    """
    if use_web_app is None:
        use_web_app = not is_running_in_notebook()

    vc = assemble_neuroglancer_config(*args, use_web_app=use_web_app, **kwargs)

    if use_web_app:
        web_app_port = find_free_port()
        vitessce_url = vc.web_app(port=web_app_port, open=False)
        if len(vitessce_url) > VITESSCE_WEB_APP_URL_WARN_LENGTH:
            warnings.warn(
                f"The generated vitessce.io URL is {len(vitessce_url)} characters long, "
                f"which exceeds the {VITESSCE_WEB_APP_URL_WARN_LENGTH}-character heuristic threshold.",
                stacklevel=2,
            )
        webbrowser.open(vitessce_url)
        input("Server running -- press Enter to stop...\n")
        return vc
    return vc.widget()