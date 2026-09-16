"""Convert the actual learned rollout recording into a portable MP4."""
from pathlib import Path
import subprocess
import imageio_ffmpeg

root=Path(__file__).resolve().parents[1]/'artifacts'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(root/'rollout.gif'),
    '-movflags','+faststart','-pix_fmt','yuv420p',str(root/'TableTogether-learned-demo.mp4')],check=True)
