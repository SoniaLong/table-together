# TableTogether

**A trained vision-language policy for dual-arm table setting in MuJoCo, deployed
locally through Intel OpenVINO.**

Two simulated SO-101 arms place a cup to the right and a utensil to the left of an
existing plate. The user selects cup-first or utensil-first. The web app displays
live simulator frames and explicit success/failure evidence.

This is a constrained research prototype: simple tableware proxies, two instruction
templates and an eighteen-keyframe horizon. It is not a foundation VLA, general
robotic planner, or a claim of completed Series 2/3 hardware validation.

## Quick start

Python 3.10, Windows 11 x64 or supported Ubuntu with a working OpenGL context:

```text
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -r robotics/requirements.txt -r requirements-hardware.txt
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python robotics/app.py
```

Open http://127.0.0.1:4180 and choose trained policy. Exact supported instructions:

- `Set the table. Start with the cup.`
- `Set the table. Start with the utensil.`

The learned checkpoint, OpenVINO exports and licensed SO-101 assets are included.
If assets are absent, run `python robotics/setup_assets.py`. On headless Linux use
`MUJOCO_GL=egl` with working drivers or OSMesa. The optional expert baseline needs
`python hardware/setup.py` to download its separate language model.

## Training and evaluation

See [training documentation](robotics/learning/README.md),
[measured results](robotics/learning/RESULTS.md),
[robustness definitions](robotics/ROBUSTNESS.md), and
[hardware compatibility assessment](robotics/COMPATIBILITY.md).

```text
python robotics/learning/rollout.py --runtime openvino --start 600 --count 10 --label reproduced
python robotics/learning/robustness.py --start 700 --count 10
python robotics/learning/export_openvino.py
python -m pip install imageio-ffmpeg==0.6.0
python robotics/learning/ten_seed_video.py --start 600
```

Recorded demonstration: [all ten seeds](robotics/learning/evaluations/ten-seed-final/TableTogether-ten-seeds.mp4).
Reports preserve failures. No expert fallback is used in learned rollouts. The
ground-truth object state is used only by the evaluator, not the policy.

## Architecture

RGB CNN + ordered word encoder + twelve joint observations + action index predict
joint targets and durations. Demonstration-derived means and learned residuals form
a strong trajectory prior. MuJoCo actuators interpolate targets and real simulated
contacts move objects; no grasp welds or object teleports. The arms act sequentially
in complementary workspaces. This does not demonstrate a physical hand-off.

CPU deployment is supported in principle on Core Ultra Series 2/3 according to Intel's
OpenVINO support matrix. Measured runs here are on Series 1. A target-machine run is
still required for qualifying hardware evidence. NPU/GPU model execution is not claimed.

## Attribution

SO-101 assets: Google DeepMind MuJoCo Menagerie / The Robot Studio, Apache-2.0,
revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`. Retain the included asset LICENSE.
Fingertip friction was changed to 2. MuJoCo, PyTorch and OpenVINO retain their own licenses.
