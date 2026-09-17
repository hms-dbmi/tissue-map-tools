"""
Tests for neuroglancer_config_builder. Mesh-id discovery (_add_segmentation
with segments=None) is exercised against a real minimal CloudVolume
precomputed dataset written to tmp_path, following the same
write-real-data-then-read-it-back approach as
tests/data_model/test_sharded_annotations.py, rather than mocking
CloudVolume. Everything else builds real VitessceConfig objects and inspects
the resulting dict directly.
"""

from cloudvolume import CloudVolume
from vitessce import CoordinationType as ct

from tissue_map_tools.vitessce_configs.layer_specs import (
    SegmentationLayerSpec,
    AnnotationLayerSpec,
    TabularObsSpec,
)
from tissue_map_tools.vitessce_configs.neuroglancer_config_builder import (
    _add_segmentation,
    _add_annotation,
    _add_tabular,
    assemble_neuroglancer_config,
)


class _FakeDataset:
    """Minimal stand-in for VitessceConfigDataset: just records add_object calls."""

    def __init__(self):
        self.objects = []

    def add_object(self, obj):
        self.objects.append(obj)


def make_minimal_precomputed_segmentation(root_path, mesh_ids):
    """Write a minimal real precomputed segmentation root with an unsharded
    mesh index (no shard files), so get_ids_from_mesh_files' fallback branch
    (data_path.glob("*.index")) has real ids to find."""
    cv = CloudVolume(
        cloudpath=f"file://{root_path}",
        info=CloudVolume.create_new_info(
            num_channels=1,
            layer_type="segmentation",
            data_type="uint32",
            encoding="raw",
            resolution=[1, 1, 1],
            voxel_offset=[0, 0, 0],
            chunk_size=[64, 64, 64],
            volume_size=[64, 64, 64],
            mesh="mesh",
        ),
    )
    cv.commit_info()
    mesh_dir = root_path / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for seg_id in mesh_ids:
        (mesh_dir / f"{seg_id}.index").write_bytes(b"")
    return cv


def test_add_segmentation_with_explicit_segments_generates_single_obs_set():
    dataset = _FakeDataset()
    spec = SegmentationLayerSpec(file_uid="seg", data_url="https://example.org/seg", segments=[1, 2, 3])

    channel, added_obs_sets = _add_segmentation(dataset, spec, use_web_app=False, skip_segment_discovery=False)

    assert added_obs_sets is True
    assert len(dataset.objects) == 2  # segmentation wrapper + auto-generated obsSets CsvWrapper
    segmentation_channel = channel["segmentationChannel"].value[0]
    assert segmentation_channel[ct.OBS_COLOR_ENCODING] == "cellSetSelection"
    assert channel["fileUid"] == "seg"
    assert channel["spatialLayerLabel"] == "seg"


def test_add_segmentation_discovers_ids_from_real_precomputed_root(tmp_path):
    """Regression test for the two real bugs found while building this: the
    hardcoded '"mesh"' subpath guess, and pathlib.Path mangling a remote URL
    (get_ids_from_mesh_files) -- covered here for the local, filesystem-glob
    fallback branch specifically."""
    root = tmp_path / "precomputed_root"
    root.mkdir()
    make_minimal_precomputed_segmentation(root, mesh_ids=[1, 2, 5])

    dataset = _FakeDataset()
    spec = SegmentationLayerSpec(file_uid="seg", data_path=str(root))

    channel, added_obs_sets = _add_segmentation(dataset, spec, use_web_app=False, skip_segment_discovery=False)

    assert added_obs_sets is True
    segmentation_channel = channel["segmentationChannel"].value[0]
    assert segmentation_channel[ct.OBS_COLOR_ENCODING] == "cellSetSelection"


def test_add_segmentation_skip_segment_discovery_true_skips_even_with_no_segments(tmp_path):
    """On-demand mesh loading: with skip_segment_discovery=True, no obsSets
    CSV should be generated even though a discoverable mesh root exists."""
    root = tmp_path / "precomputed_root"
    root.mkdir()
    make_minimal_precomputed_segmentation(root, mesh_ids=[1, 2, 5])

    dataset = _FakeDataset()
    spec = SegmentationLayerSpec(file_uid="seg", data_path=str(root), obs_color_encoding="geneSelection")

    channel, added_obs_sets = _add_segmentation(dataset, spec, use_web_app=False, skip_segment_discovery=True)

    assert added_obs_sets is False
    assert len(dataset.objects) == 1  # only the segmentation wrapper, no discovery, no CSV
    segmentation_channel = channel["segmentationChannel"].value[0]
    assert segmentation_channel[ct.OBS_COLOR_ENCODING] == "geneSelection"


def test_add_segmentation_user_supplied_obs_sets_csv_skips_auto_generation():
    dataset = _FakeDataset()
    obs_sets_csv = TabularObsSpec(
        data_type="obsSets",
        csv_url="https://example.org/sets.csv",
        options={"obsIndex": "id", "obsSets": [{"name": "Cell Types", "column": "phenotype"}]},
        coordination_values={"obsType": "cell"},
    )
    spec = SegmentationLayerSpec(
        file_uid="seg", data_url="https://example.org/seg",
        segments=[1], obs_sets_csv=obs_sets_csv,
    )

    channel, added_obs_sets = _add_segmentation(dataset, spec, use_web_app=False, skip_segment_discovery=False)

    assert added_obs_sets is True
    segmentation_channel = channel["segmentationChannel"].value[0]
    assert segmentation_channel[ct.OBS_COLOR_ENCODING] == "cellSetSelection"


def test_add_annotation_channel_includes_feature_type():
    """Regression test: featureType must be on the per-layer channel dict,
    not just the file's coordinationValues, or gene-selection coloring
    silently fails to update."""
    dataset = _FakeDataset()
    spec = AnnotationLayerSpec(
        file_uid="points", data_url="https://example.org/points",
        obs_type="point", feature_type="gene",
    )

    channel = _add_annotation(dataset, spec)

    assert channel["featureType"] == "gene"
    assert channel["obsType"] == "point"
    assert ct.FEATURE_VALUE_COLORMAP not in channel
    assert ct.FEATURE_SELECTION not in channel


def test_add_annotation_includes_feature_selection_when_set():
    """The raw channel dict this function returns is keyed by CoordinationType
    enum members (e.g. ct.FEATURE_SELECTION), not plain strings -- Vitessce's
    own CL()/link_views_by_dict machinery normalizes these to their string
    form later (see test_feature_selection_reaches_serialized_config_correctly)."""
    dataset = _FakeDataset()
    spec = AnnotationLayerSpec(
        file_uid="points", data_url="https://example.org/points",
        feature_selection=["Ada"], feature_value_colormap="plasma",
        feature_value_colormap_range=(0.0, 1.0),
    )

    channel = _add_annotation(dataset, spec)

    assert channel[ct.FEATURE_SELECTION] == ["Ada"]
    assert channel["featureFilterMode"] == "featureSelection"
    assert channel[ct.FEATURE_VALUE_COLORMAP] == "plasma"


def test_feature_selection_reaches_serialized_config_correctly():
    """End-to-end check that the enum-keyed intermediate dict above
    serializes to the right plain-string key in the final config."""
    vc = assemble_neuroglancer_config(
        annotations=[AnnotationLayerSpec(
            file_uid="pts", data_url="https://example.org/pts", feature_selection=["Ada"],
        )],
    )
    coordination_space = vc.to_dict()["coordinationSpace"]
    assert list(coordination_space["featureSelection"].values()) == [["Ada"]]


def test_add_tabular_omits_empty_options():
    """Regression test: an empty options dict must become None, not {}, or
    config validation rejects file types whose schema expects options to be
    entirely absent. CsvWrapper exposes no public accessor for this, so we
    check its internal _options rather than the config's rendered dict."""
    dataset = _FakeDataset()
    spec = TabularObsSpec(data_type="obsFeatureMatrix", csv_url="https://example.org/x.csv")

    _add_tabular(dataset, spec)

    wrapper = dataset.objects[0]
    assert wrapper._options is None


def test_build_neuroglancer_config_adds_obs_sets_view_when_needed():
    vc = assemble_neuroglancer_config(
        segmentations=[SegmentationLayerSpec(file_uid="seg", data_url="https://example.org/seg", segments=[1, 2])],
    )
    components = [v["component"] for v in vc.to_dict(base_url="http://localhost:8000")["layout"]]
    assert "obsSets" in components


def test_build_neuroglancer_config_omits_obs_sets_view_when_not_needed():
    vc = assemble_neuroglancer_config(
        segmentations=[SegmentationLayerSpec(
            file_uid="seg", data_url="https://example.org/seg", segments=[1, 2],
            auto_generate_obs_sets=False,
        )],
    )
    components = [v["component"] for v in vc.to_dict(base_url="http://localhost:8000")["layout"]]
    assert "obsSets" not in components


def test_build_neuroglancer_config_skips_discovery_for_all_segmentations_when_threshold_set(tmp_path):
    """meshLoadProjectionScaleThreshold is a view-level Neuroglancer prop, not
    per-layer -- setting it must skip mesh-id discovery config-wide."""
    root = tmp_path / "precomputed_root"
    root.mkdir()
    make_minimal_precomputed_segmentation(root, mesh_ids=[1, 2])

    vc = assemble_neuroglancer_config(
        segmentations=[SegmentationLayerSpec(file_uid="seg", data_path=str(root))],
        mesh_load_projection_scale_threshold=1200,
    )
    components = [v["component"] for v in vc.to_dict(base_url="http://localhost:8000")["layout"]]
    assert "obsSets" not in components