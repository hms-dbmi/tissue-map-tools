from tissue_map_tools.view import (  # noqa: F401
    view_precomputed_in_napari,
    view_precomputed_in_neuroglancer,
)
from pathlib import Path

dataset_path = Path(__file__).parent.parent.parent / "data" / "merfish_mouse_ileum"
out_path = dataset_path / "out"
precomputed_path = out_path / "merfish_mouse_ileum_precomputed"


viewer = view_precomputed_in_neuroglancer(
    data_path=str(precomputed_path),
)
# view_precomputed_in_napari(
#     data_path="/Users/macbook/Desktop/moffitt_precomputed",
#     show_meshes=False,
#     show_raster=True,
#     show_points=True,
# )
