from pathlib import Path
from tissue_map_tools.data_model.annotations_utils import parse_annotations

out_path = Path(__file__).parent.parent.parent / "out"
precomputed_path = out_path / "merfish_mouse_ileum_precomputed"

if __name__ == "__main__":
    data_path = precomputed_path

    # parse raster

    # parse meshes
    # TODO: draft code to be ported
    # from cloudvolume import CloudVolume
    # from tissue_map_tools.utils import get_ids_from_mesh_files
    #
    # cv = CloudVolume(str(data_path))
    # mesh_file = cv.info["mesh"]
    # mesh_ids = get_ids_from_mesh_files(
    #     data_path=data_path / mesh_file, root_data_path=data_path
    # )
    # ##
    # index = 6
    # for lod in range(3):
    #     mesh = cv.mesh.get(segids=mesh_ids[index], lod=lod)[mesh_ids[index]]
    #     with open(f"mesh_lod{lod}.obj", "wb") as f:
    #         f.write(mesh.to_obj())

    # parse annotations
    df_annotations = parse_annotations(data_path)
    print(df_annotations)
    print(df_annotations["x"].max())
    print(df_annotations["y"].max())
    print(df_annotations["z"].max())
    print(df_annotations["x"].min())
    print(df_annotations["y"].min())
    print(df_annotations["z"].min())
    print(df_annotations.iloc[0])

    pass
