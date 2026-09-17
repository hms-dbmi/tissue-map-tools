from tissue_map_tools.vitessce_configs.layer_specs import (
    SegmentationLayerSpec,
    AnnotationLayerSpec,
    TabularObsSpec,
    SpatialDataObsSpec,
)


def test_segmentation_layer_spec_defaults():
    """A spec with only file_uid set should carry sensible defaults."""
    spec = SegmentationLayerSpec(file_uid="segmentation")
    assert spec.data_path is None
    assert spec.data_url is None
    assert spec.obs_type == "cell"
    assert spec.segments is None
    assert spec.obs_sets_csv is None
    assert spec.auto_generate_obs_sets is True
    assert spec.obs_color_encoding == "cellSetSelection"
    assert spec.spatial_channel_color is None


def test_annotation_layer_spec_defaults():
    spec = AnnotationLayerSpec(file_uid="points")
    assert spec.obs_type == "cell"
    assert spec.feature_type == "gene"
    assert spec.obs_color_encoding == "geneSelection"
    assert spec.spatial_point_stroke_width == 0.1
    assert spec.feature_selection is None


def test_tabular_obs_spec_holds_csv_options_verbatim():
    spec = TabularObsSpec(
        data_type="obsSets",
        csv_url="https://example.org/sets.csv",
        options={"obsIndex": "id", "obsSets": [{"name": "Cell Types", "column": "phenotype"}]},
        coordination_values={"obsType": "cell"},
    )
    assert spec.csv_path is None
    assert spec.options["obsSets"][0]["name"] == "Cell Types"


def test_spatial_data_obs_spec_defaults():
    spec = SpatialDataObsSpec(sdata_url="https://example.org/data.sdata.zarr")
    assert spec.table_path == "tables/table"
    assert spec.obs_type == "cell"
    assert spec.obs_feature_matrix_path is None
    assert spec.obs_set_paths is None