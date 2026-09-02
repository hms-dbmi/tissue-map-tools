from pathlib import Path

from tissue_map_tools.view import view_precomputed_in_vitessce

##
# Path to a precomputed dataset produced by one of the other examples
# (e.g. examples/invasive/ome_tiff_to_sharded_meshes.py), containing a
# segmentation + meshes, and optionally point annotations alongside it.
precomputed_path = Path(__file__).parent.parent.parent / "data" / "melanoma_precomputed"

if not precomputed_path.exists():
    raise FileNotFoundError(
        f"{precomputed_path} does not exist. Run the invasive mesh-generation "
        "example first, or point precomputed_path at your own precomputed data."
    )

view_precomputed_in_vitessce(
    data_path=str(precomputed_path),
    show_meshes=True,
    show_annotations=True,
    port=10001,
    host="localhost",
    name="Melanoma",
    host_local_data=True,
    segments=[612, 3351, 4328, 6531, 8446],
    use_web_app=True,
)