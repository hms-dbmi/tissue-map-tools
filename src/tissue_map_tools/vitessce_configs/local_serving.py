import threading
import time
from cloudvolume import CloudVolume
from tissue_map_tools.utils import find_free_port


def serve_local_precomputed(local_path: str) -> str:
    """
    Serve a local Neuroglancer precomputed root (segmentation+meshes, or a
    point-annotations directory) over HTTP via CloudVolume's built-in viewer,
    matching the host_local_data pattern in tissue_map_tools.view.view_precomputed_in_vitessce.
    """
    cv = CloudVolume(cloudpath=local_path)
    port = find_free_port()
    thread = threading.Thread(target=cv.viewer, kwargs={"port": port}, daemon=True)
    thread.start()
    time.sleep(1)
    return f"http://localhost:{port}"


def resolve_url(spec) -> str:
    if spec.local_path and spec.data_url:
        raise ValueError(f"{spec.file_uid}: set local_path OR data_url, not both")
    if spec.local_path:
        return serve_local_precomputed(spec.local_path)
    return spec.data_url