import webbrowser
import warnings
from typing import Any
import colorsys
from pathlib import Path
from cloudvolume import CloudVolume

from .layer_specs import SegmentationLayerSpec, AnnotationLayerSpec, TabularObsSpec, SpatialDataObsSpec
from tissue_map_tools.shard_util import get_ids_from_mesh_files
from tissue_map_tools.data_model.annotations import find_annotations_from_cloud_volume
from tissue_map_tools.view import compute_initial_camera_state
from vitessce import (
    VitessceConfig, CoordinationLevel as CL, CoordinationType as ct,
    DataType as dt, hconcat, vconcat, get_initial_coordination_scope_prefix,
    CsvWrapper, ObsSegmentationsNgPrecomputedWrapper, ObsPointsNgAnnotationsWrapper,
    make_ids_csv_data_url, make_colors_csv_data_url,
)

from tissue_map_tools.shard_util import get_ids_from_mesh_files
from vitessce import make_ids_csv_data_url, make_colors_csv_data_url
from tissue_map_tools.utils import is_running_in_notebook, find_free_port

VITESSCE_WEB_APP_URL_WARN_LENGTH = 8_000

def _add_segmentation(dataset, spec: SegmentationLayerSpec, use_web_app: bool):
    resolved_ids = spec.segments
    if resolved_ids is None:
        cv_path = spec.local_path or spec.data_url
        if cv_path:
            cv = CloudVolume(cloudpath=cv_path)
            mesh_subpath = cv.meta.info.get("mesh")
            if mesh_subpath is not None:
                resolved_ids = get_ids_from_mesh_files(
                    root_data_path=cv_path,
                    data_path=Path(cv_path) / mesh_subpath,
                )
    resolved_ids = [str(i) for i in (resolved_ids or []) if str(i) != "0"]
    dataset.add_object(ObsSegmentationsNgPrecomputedWrapper(
        data_path=spec.local_path, 
        data_url=spec.data_url,
        coordination_values={"fileUid": spec.file_uid},
        options=spec.options or None,
    ))
    added_obs_sets = False
    if spec.obs_sets_csv is not None:
        _add_tabular(dataset, spec.obs_sets_csv)
        added_obs_sets = True
    elif resolved_ids and spec.auto_generate_obs_sets:
        set_name = spec.obs_set_name or spec.label or spec.file_uid
        dataset.add_object(CsvWrapper(
            csv_url=make_obs_set_csv_data_url(resolved_ids, set_name, for_web_app=use_web_app),
            data_type="obsSets",
            options={"obsIndex": "id", "obsSets": [{"name": set_name, "column": set_name}]},
            coordination_values={"obsType": spec.obs_type},
        ))
        added_obs_sets = True

    channel = {"obsType": spec.obs_type, "spatialChannelVisible": True}
    if added_obs_sets:
        channel[ct.OBS_COLOR_ENCODING] = "cellSetSelection"

    channel_dict = {
        "fileUid": spec.file_uid,
        "spatialLayerOpacity": 1,
        "spatialLayerVisible": True,
        "spatialLayerLabel": spec.label or spec.file_uid,
        "segmentationChannel": CL([channel]),
    }
    return channel_dict, added_obs_sets

def _add_annotation(dataset, spec: AnnotationLayerSpec):
    dataset.add_object(ObsPointsNgAnnotationsWrapper(
        data_path=spec.local_path,
        data_url=spec.data_url,
        coordination_values={
            "fileUid": spec.file_uid,
            "obsType": spec.obs_type,
            "featureType": spec.feature_type,
        },
        options=spec.options or None,
    ))
    return {
        "fileUid": spec.file_uid,
        "obsType": spec.obs_type,
        "spatialLayerOpacity": 1,
        "spatialLayerVisible": True,
        "spatialLayerColor": spec.color,
        "spatialPointStrokeWidth": spec.stroke_width,
        "spatialLayerLabel": spec.label or spec.file_uid,
        ct.OBS_COLOR_ENCODING: spec.color_encoding,
        ct.FEATURE_VALUE_COLORMAP: spec.feature_value_colormap,
        ct.FEATURE_VALUE_COLORMAP_RANGE: spec.feature_value_colormap_range,
        ct.FEATURE_SELECTION: spec.feature_selection,
        "featureFilterMode": "featureSelection" if spec.feature_selection else None,
    }


def _add_tabular(dataset, spec: TabularObsSpec):
    dataset.add_object(CsvWrapper(
        csv_path=spec.csv_path,
        csv_url=spec.csv_url,
        data_type=spec.data_type,
        options=spec.options,
        coordination_values=spec.coordination_values,
    ))

from vitessce import SpatialDataWrapper

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

    Mirrors vitessce.make_ids_csv_data_url / make_colors_csv_data_url's
    pattern exactly (csv writer -> quote -> data:text/csv,...) — there is no
    built-in equivalent in vitessce-python today, though make_ids_csv_data_url's
    own docstring anticipates an obsSets.csv use case like this one.
    """
    import csv, io
    from urllib.parse import quote

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", set_column])
    writer.writerows([[i, set_name] for i in ids])
    encoded = quote(buf.getvalue())
    if for_web_app:
        encoded = quote(encoded, safe="")
    return f"data:text/csv,{encoded}"

def build_neuroglancer_config(
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
    use_web_app: bool | None = None,
    spatialdata_obs: list[SpatialDataObsSpec] = (),
):
    """
    Build a Vitessce Neuroglancer config from one or more segmentation, point
    annotation, and tabular obs (CSV / spatialdata.zarr) layers, and return it
    as a ready viewer - generalized to multiple layers of each kind.

    Local vs. remote layers are decided per-spec (`local_path` vs. `data_url`
    on each `SegmentationLayerSpec`/`AnnotationLayerSpec`) rather than by a
    single flag for the whole config, since a multi-layer config may mix
    local and already-remote sources.

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
        CSV/spatialdata.zarr obs sources (obsSets, obsEmbedding, obsFeatureMatrix)
        to add alongside the spatial layers, e.g. for a gene expression matrix or
        cell-type sets. Empty by default.
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
        across the Neuroglancer and layerControllerBeta views. Defaults match
        `view_precomputed_in_vitessce`'s hardcoded values (3D rendering, camera
        centered at the origin with no rotation) — override any of these to
        start the view in a different state.
    use_web_app
        If None (default), auto-detected: True when running outside a Jupyter
        notebook (plain script or terminal), False when running inside one. Set
        explicitly to override this — e.g. force True in a notebook if you
        specifically want the `vitessce.io` browser tab instead of the inline widget.

    Returns
    -------
    A Vitessce widget (via `VitessceConfig.widget()`) if `use_web_app=False`, or
    the `VitessceConfig` object itself if `use_web_app=True` (after opening a
    browser tab and blocking until the user presses Enter).
    """
    if use_web_app is None:
        use_web_app = not is_running_in_notebook()

    vc = VitessceConfig(schema_version=schema_version, name=name)
    dataset = vc.add_dataset(name=name)

    for t in tabular_obs:
        _add_tabular(dataset, t)
    spatialdata_added_obs_sets = [_add_spatialdata(dataset, s) for s in spatialdata_obs]
    seg_results = [_add_segmentation(dataset, s, use_web_app) for s in segmentations]
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

    if use_web_app:
        web_app_port = find_free_port()
        vitessce_url = vc.web_app(port=web_app_port, open=False)
        if len(vitessce_url) > VITESSCE_WEB_APP_URL_WARN_LENGTH:
            warnings.warn(
                f"The generated vitessce.io URL is {len(vitessce_url)} characters long, "
                f"which exceeds the {VITESSCE_WEB_APP_URL_WARN_LENGTH}-character heuristic "
                "threshold. This usually happens with many auto-discovered segments, "
                "since every segment id/color is embedded inline in the URL. Pass an "
                "explicit, smaller `segments` list on the SegmentationLayerSpec, or "
                "use `use_web_app=False` for the notebook-inline widget instead, which "
                "serves data over local HTTP rather than embedding it in the URL.",
                stacklevel=2,
            )
        webbrowser.open(vitessce_url)
        input("Server running -- press Enter to stop...\n")
        return vc
    return vc.widget()