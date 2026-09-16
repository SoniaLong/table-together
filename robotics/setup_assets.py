"""Fetch only the pinned upstream robot asset directory; no system modifications."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
destination=ROOT/'vendor/mujoco_menagerie'
revision='8161bba264d7fa7c99ca301e91e7fb44737676ad'
def git(*args):subprocess.run(['git',*map(str,args)],check=True)
if not destination.exists():
    git('clone','--filter=blob:none','--no-checkout','https://github.com/google-deepmind/mujoco_menagerie.git',destination)
git('-C',destination,'sparse-checkout','set','robotstudio_so101')
git('-C',destination,'checkout','--detach',revision)
print('SO-101 assets ready. License:',destination/'robotstudio_so101/LICENSE')
