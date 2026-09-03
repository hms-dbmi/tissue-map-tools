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
    show_annotations=False,
    port=10001,
    host="localhost",
    name="Melanoma",
    host_local_data=True,
    segments=[612, 3351, 4328, 6531, 8446],
    segment_colors = { 
        612: '#d74242',
      3351: '#b9d742',
      4328: '#42d77d',
      6531: '#427dd7',
      8446: '#b942d7'
    },
    use_web_app=True,
    initial_camera_state={
            "position": [5217273.5, 554404.125, 97352.421875],
            "projectionScale": 1024,
            "projectionOrientation": [
            -0.636204183101654,
            -0.5028395652770996,
            0.5443811416625977,
            0.2145828753709793,
        ],
    }
)
