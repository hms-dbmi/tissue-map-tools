from dataclasses import dataclass, field


@dataclass
class SegmentationLayerSpec:
    file_uid: str
    local_path: str | None = None
    data_url: str | None = None
    obs_type: str = "cell"
    label: str | None = None
    options: dict = field(default_factory=dict)          # e.g. {"dimensions": {...}, "segments": [...], "segmentColors": {...}}
    channel_color: list[int] | None = None
    color_encoding: str = "obsColors"                     # OBS_COLOR_ENCODING values
    feature_type: str | None = None
    feature_value_type: str | None = None
    feature_selection: list[str] | None = None
    feature_value_colormap: str | None = None
    feature_value_colormap_range: tuple[float, float] | None = None


@dataclass
class AnnotationLayerSpec:
    file_uid: str
    local_path: str | None = None
    data_url: str | None = None
    obs_type: str = "cell"
    feature_type: str = "gene"
    label: str | None = None
    color: list[int] | None = None
    color_encoding: str = "geneSelection"
    feature_selection: list[str] | None = None
    feature_value_colormap: str | None = None
    feature_value_colormap_range: tuple[float, float] | None = None
    stroke_width: float = 0.2
    options: dict = field(default_factory=dict)           # featureIndexProp, pointIndexProp, transform, quantitativeColorProp, quantitativeColorMax, projectionAnnotationSpacing, ...


@dataclass
class TabularObsSpec:
    """One CSV/JSON/spatialdata.zarr source of obsSets / obsEmbedding / obsFeatureMatrix."""
    data_type: str                 # use vitessce.constants.DataType, e.g. dt.OBS_SETS
    csv_url: str | None = None
    csv_path: str | None = None
    options: dict = field(default_factory=dict)
    coordination_values: dict = field(default_factory=dict)