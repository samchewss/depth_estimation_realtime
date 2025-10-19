
from huggingface_hub import snapshot_download
import os
target = os.path.join("models", "Depth-Anything-V2-Small-hf")  # o "Depth-Anything-V2-Base-hf"
snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Small-hf", local_dir=target, local_dir_use_symlinks=False)
print("Descarga completa en:", target)
