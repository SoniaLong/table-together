"""Download the exact tested model revision to the project-local model directory."""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov'
REVISION = '339876973306da551969e2d88991b7053be2029c'
target = Path(__file__).resolve().parents[1] / 'models' / 'qwen'
snapshot_download(REPO, revision=REVISION, local_dir=str(target))
(target / 'REVISION').write_text(REVISION)
print('Model ready:', target)
