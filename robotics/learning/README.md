# Learning a table-setting policy

This is a small task-specific vision-language **keyframe imitation policy**, trained
from scratch on this simulator. It is not a pretrained foundation VLA and does not
implement ACT. The low-level actuator interpolation remains conventional control;
the learned model selects the joint setpoints and durations during rollout.

## Inputs and actions

- 128×96 overhead RGB image, directly processed by a learned CNN.
- Twelve measured joint positions.
- Ordered instruction word tokens, processed by a learned language encoder.
- Ordinal position in the 18-keyframe action history.
- Output: twelve joint targets plus a motion duration.

Two training instructions request different orders: cup first or utensil first.
The demonstration mean and variation for each keyframe are estimated from the
training data; the neural policy learns language selection and observation-dependent
residual actions. There is no inverse-kinematics call, color-segmentation target, or
scripted expert fallback in `rollout.py`. Ground-truth object poses appear only in
evaluation. The 18-keyframe horizon is fixed, so this does not establish arbitrary
long-horizon reasoning or general natural-language understanding.

## Reproduction

Install PyTorch CPU into the existing environment:

```text
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python robotics/learning/collect.py --split train --start 100 --count 24
python robotics/learning/collect.py --split train --start 400 --count 16 --jitter .012
python robotics/learning/collect.py --split validation --start 200 --count 4
python robotics/learning/train.py --iterations 5000 --version 2 --warmstart robotics/learning/checkpoints/original.pt --output reproduced-training
python robotics/learning/export_openvino.py
python robotics/learning/rollout.py --runtime openvino --label test-openvino-fp32
python robotics/learning/rollout.py --start 300 --count 10 --label test
python robotics/learning/rollout.py --start 300 --count 10 --task 1 --label reverse-order
python robotics/learning/rollout.py --start 300 --count 10 --ablation blank_image --label blank-image
```

To record an illustrative rollout and convert it to MP4:

```text
python -m pip install imageio-ffmpeg==0.6.0
```

Run learned mode in the app (it writes artifacts/rollout.gif), then run
`python robotics/learning/package_demo.py`. This video is a selected single episode,
not the required ten-seed competition demonstration.

Only successful expert episodes with 18 actions enter imitation training; failed
collection attempts remain in the manifests. Each episode stores RGB observations,
joint states, actions, seed and language-task identity in an NPZ file. This native
format is not claimed to be a LeRobot dataset.

Training/validation splits use different seeds. Policy checkpoint selection uses
validation action error; final task success must come from independent MuJoCo
rollouts, not the supervised loss. Reports must identify checkpoint hashes and must
not combine results across changing models as one benchmark.

The standard evaluation randomizes positions by ±8 mm. Separate robustness profiles
vary lighting, background, mass, friction and horizontal dimensions; see ../ROBUSTNESS.md.
General recovery, unfamiliar shapes and qualifying Series 2/3 validation remain unfinished.

## Frozen training result

75 successful training episodes (1,350 keyframes) out of 80 collection attempts;
8 validation episodes (144 keyframes). Training seeds 100–123 and 400–415, validation
200–203, final test/video 600–609. Both instruction orders were collected. Warm-start
CPU training took 288.08 seconds for 5,000 iterations; validation normalized error
selected iteration 4,200. Validation joint MAE: 0.001933 radians versus 0.002598 for
the original. This supervised metric is not a task-success rate.

Checkpoint SHA256: `f2f60dde4d383f845b91e9033afe51a691ece023a1bf53a6e29452e53dbb6bc3`.

Complete validation rollouts improved from 6/8 to 7/8. The revised architecture has
a separate residual output head per keyframe, initialized from the original learned
weights. Training applies brightness variation (0.85–1.15) and 0.001-radian joint
observation noise. The scratch candidate did not improve validation and was rejected.
See EVALUATION_PLAN.md, checkpoints/experiment.json and RESULTS.md. Original test and
camera-ablation results are retained in RESULTS-original.md; they do not describe the
revised checkpoint. The trajectory prior remains substantial.

The app defaults to learned OpenVINO inference. It accepts the two exact instruction
templates above and reports unsupported text as an error. The separate baseline mode
uses Qwen and classical control; it is never an automatic learned-policy fallback.
