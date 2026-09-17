from dataclasses import dataclass, field


@dataclass
class TabularObsSpec:
    """One CSV/JSON/spatialdata.zarr source of obsSets / obsEmbedding / obsFeatureMatrix."""
    data_type: str                 # use vitessce.constants.DataType, e.g. dt.OBS_SETS
    csv_url: str | None = None
    csv_path: str | None = None
    options: dict = field(default_factory=dict)
    coordination_values: dict = field(default_factory=dict)
    
@dataclass
class SegmentationLayerSpec:
    file_uid: str
    data_path: str | None = None
    data_url: str | None = None
    obs_type: str = "cell"
    spatial_layer_label: str | None = None
    options: dict = field(default_factory=dict)          # e.g. {"dimensions": {...}}
    spatial_channel_color: list[int] | None = None
    obs_color_encoding: str = "obsColors"                     # OBS_COLOR_ENCODING values
    feature_type: str | None = None
    feature_value_type: str | None = None
    feature_selection: list[str] | None = None
    feature_value_colormap: str | None = None
    feature_value_colormap_range: tuple[float, float] | None = None
    segments: list[int] | list[str] | None = None
    """Specific segment IDs to select and color. If None, all real object IDs
    are auto-discovered from the mesh shard files (via get_ids_from_mesh_files)."""
    # TODO: Uncomment when upstream bug is resolved (https://github.com/vitessce/vitessce-python/issues/517)
    # segment_colors: dict[str, str] | None = None
    # """Optional dict mapping segment ID (as a string) to a hex color. If not
    # provided, a default HSV rainbow color is generated per segment."""
    obs_sets_csv: TabularObsSpec | None = None
    """User-supplied obsSets TabularObsSpec for this layer's set/coloring data.
    If None (default) and auto_generate_obs_sets is True, one is auto-generated
    instead, assigning every resolved segment id to a single set named
    obs_set_name."""
    auto_generate_obs_sets: bool = True
    """Whether to auto-generate the single-set obsSets CSV described above when
    obs_sets_csv is not provided. Set False when real sets come from elsewhere
    (e.g. a spatialdata_obs entry or csv) — otherwise the auto-generated set is
    redundant alongside the real ones."""
    obs_set_name: str = "Cell Sets"
    obs_color_encoding: str = "cellSetSelection"
    

@dataclass
class AnnotationLayerSpec:
    file_uid: str
    data_path: str | None = None
    data_url: str | None = None
    obs_type: str = "cell"
    feature_type: str = "gene"
    spatial_layer_label: str | None = None
    spatial_layer_color: list[int] | None = None
    obs_color_encoding: str = "geneSelection"
    feature_selection: list[str] | None = None
    feature_value_colormap: str | None = None
    feature_value_colormap_range: tuple[float, float] | None = None
    spatial_point_stroke_width: float = 0.1
    options: dict = field(default_factory=dict)           # featureIndexProp, pointIndexProp, transform, quantitativeColorProp, quantitativeColorMax, projectionAnnotationSpacing, ...


@dataclass
class SpatialDataObsSpec:
    """
    One obs source (feature matrix and/or cell sets) read from a single table
    within a spatialdata.zarr store, via vitessce.SpatialDataWrapper.

    SpatialDataWrapper takes one `table_path` — if the feature matrix and the
    sets live under different tables in the same store (as in the MERFISH
    dataset), use one SpatialDataObsSpec per table, all sharing the same
    sdata_path/sdata_url and obs_type.
    """
    sdata_path: str | None = None
    sdata_url: str | None = None
    table_path: str = "tables/table"
    obs_type: str = "cell"
    feature_type: str | None = None
    obs_feature_matrix_path: str | None = None    # e.g. "X"
    obs_set_paths: list[str] | None = None        # e.g. ["obs/cluster"]
    obs_set_names: list[str] | None = None        # e.g. ["Cell Types"]
    obs_feature_matrix_path: str | None = None
    """Full path to the expression matrix within the .sdata.zarr store, e.g.
    tables/gene_expression_baysor/X """