from .layer_specs import SegmentationLayerSpec, AnnotationLayerSpec, TabularObsSpec
from tissue_map_tools.shard_util import get_ids_from_mesh_files
from tissue_map_tools.data_model.annotations import find_annotations_from_cloud_volume
from tissue_map_tools.view import compute_initial_camera_state
from vitessce import (
    VitessceConfig, CoordinationLevel as CL, CoordinationType as ct,
    DataType as dt, hconcat, vconcat, get_initial_coordination_scope_prefix,
    CsvWrapper, ObsSegmentationsNgPrecomputedWrapper, ObsPointsNgAnnotationsWrapper,
    make_ids_csv_data_url, make_colors_csv_data_url,
)
from .local_serving import resolve_url

def _add_segmentation(dataset, spec: SegmentationLayerSpec):
    dataset.add_object(ObsSegmentationsNgPrecomputedWrapper(
        data_url=resolve_url(spec),
        coordination_values={"fileUid": spec.file_uid, "obsType": spec.obs_type},
        options=spec.options or None,
    ))
    channel = {
        "obsType": spec.obs_type,
        "spatialChannelVisible": True,
        "spatialChannelColor": spec.channel_color,
        ct.OBS_COLOR_ENCODING: spec.color_encoding,
    }
    if spec.color_encoding == "geneSelection":
        channel.update({
            "featureType": spec.feature_type,
            "featureValueType": spec.feature_value_type,
            ct.FEATURE_SELECTION: spec.feature_selection,
            ct.FEATURE_VALUE_COLORMAP: spec.feature_value_colormap,
            ct.FEATURE_VALUE_COLORMAP_RANGE: spec.feature_value_colormap_range,
        })
    return {
        "fileUid": spec.file_uid,
        "spatialLayerOpacity": 1,
        "spatialLayerVisible": True,
        "spatialLayerLabel": spec.label or spec.file_uid,
        "segmentationChannel": CL([channel]),
    }


def _add_annotation(dataset, spec: AnnotationLayerSpec):
    dataset.add_object(ObsPointsNgAnnotationsWrapper(
        data_url=resolve_url(spec),
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


def build_neuroglancer_config(
    name: str,
    schema_version: str,
    segmentations: list[SegmentationLayerSpec] = (),
    annotations: list[AnnotationLayerSpec] = (),
    tabular_obs: list[TabularObsSpec] = (),
    initial_camera_state: dict | None = None,
    show_axis_lines: bool | None = None,
    mesh_load_projection_scale_threshold: float | None = None,
    layer_per_feature_for_points: bool | None = None,
    extra_view_types: list[str] = (),   # e.g. [vt.FEATURE_LIST, vt.OBS_SETS, vt.SCATTERPLOT]
) -> VitessceConfig:
    vc = VitessceConfig(schema_version=schema_version, name=name)
    dataset = vc.add_dataset(name=name)

    for t in tabular_obs:
        _add_tabular(dataset, t)

    seg_channels = [_add_segmentation(dataset, s) for s in segmentations]
    point_layers = [_add_annotation(dataset, a) for a in annotations]

    ng_view = vc.add_view("neuroglancer", dataset=dataset)
    ng_props = {}
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

    extra_views = [vc.add_view(v, dataset=dataset) for v in extra_view_types]

    vc.link_views_by_dict([ng_view, lc_view], {
        "spatialRenderingMode": "3D",
        "spatialZoom": 0, "spatialTargetT": 0,
        "spatialTargetX": 0, "spatialTargetY": 0, "spatialTargetZ": 0,
        "spatialRotationX": 0, "spatialRotationY": 0, "spatialRotationZ": 0,
        "spatialRotationOrbit": 0,
    }, meta=False)

    if seg_channels:
        vc.link_views_by_dict([ng_view, lc_view],
            {"segmentationLayer": CL(seg_channels)},
            scope_prefix=get_initial_coordination_scope_prefix("A", dt.OBS_SEGMENTATIONS))

    if point_layers:
        vc.link_views_by_dict([ng_view, lc_view],
            {"pointLayer": CL(point_layers)},
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsPoints"))

    vc.layout(hconcat(ng_view, vconcat(lc_view, *extra_views)))
    return vc