# Frozen trained-policy results

Checkpoint SHA256: `c6d10bea820eda3ec00ae5987e2823f73023c605494aa8924d18f89dbbfe8f7e`.
Trained locally on Core Ultra 5 125U CPU (Series 1). No qualifying Series 2/3 run.

43 successful expert training episodes / 774 keyframes; 8 validation episodes /
144 keyframes. Training took 127.94 seconds for 5,000 iterations. Validation selected
iteration 3,600. Validation joint MAE 0.002598 radians. Collection failures remain
in the manifests. This is a small custom policy trained from scratch, not ACT or a
pretrained foundation VLA.

## Task evaluation

Every row uses the frozen checkpoint and seeds 300–309, independent of training
(100–123) and validation (200–203). Initial object XY positions vary by ±8 mm.
Shapes, lighting, mass and friction are fixed. Success requires both placements
within 30 mm, upright cup, undisturbed plate and the requested first moving arm.

| Runtime / condition | Instruction | Success |
|---|---|---|
| PyTorch FP32 | Cup first | 6/10 |
| PyTorch FP32 | Utensil first | 8/10 |
| PyTorch, camera blacked out | Cup first | 4/10 |
| OpenVINO FP32 weights / FP32 CPU compute | Cup first | 6/10 |
| OpenVINO FP16 weights / FP32 CPU compute | Cup first | 7/10 |

The two normal PyTorch conditions total 14/20. Requested first-arm order was correct
in all 20. Cup placement is the primary failure. The small camera ablation difference
does not establish broad visual generalization: the model has a strong learned
trajectory prior and a fixed eighteen-keyframe horizon. It does not recover generally
or understand unrestricted natural language. No expert fallback runs during evaluation.

FP16 weight storage is approximately half FP32. The one-episode task difference is
not evidence of general superiority; small action changes can alter contact dynamics.
The default app retains FP32. See checkpoints/openvino-benchmark.json for actual
warmup, latency, serial throughput, precision and validation parity measurements.
These are local development timings, not qualifying hardware performance. Each
evaluation summary includes all failed and successful episodes and checkpoint hash.

## Reproduce

```text
python robotics/learning/train.py --iterations 5000
python robotics/learning/export_openvino.py
python robotics/learning/rollout.py --runtime openvino --precision fp32 --label test-openvino-fp32
python robotics/learning/rollout.py --runtime openvino --precision fp16_weights --label test-openvino-fp16
```

Retraining can produce a different checkpoint on another software/hardware stack.
Use the packaged checkpoint to reproduce this frozen evaluation. The included
training JSON identifies the data files and training settings. The selected video
uses seed 301 and is illustrative; it does not replace a ten-seed demonstration.

## Remaining competition gaps

Qualifying Series 2/3 execution, broader randomization and reliability, a ten-seed
demonstration video, a published reproducible repository and final submission remain
unfinished. The local trained product must not be described as a fully compliant
or fully reliable hackathon submission.
