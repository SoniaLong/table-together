# Improved trained-policy results

Frozen checkpoint: `f2f60dde4d383f845b91e9033afe51a691ece023a1bf53a6e29452e53dbb6bc3`.

75 successful demonstrations / 1,350 keyframes, versus the original 43 / 774. Eight separate validation episodes. Warm-start training: 5,000 iterations, 288.08 seconds on local CPU.
Validation-selected iteration 4,200; joint MAE 0.001933 radians versus 0.002598 (25.6% lower).
Full validation success: 7/8 versus 6/8 on the same seeds and instruction orders. This small validation improvement is not a broad reliability guarantee.

## Fresh final evaluation

OpenVINO CPU, seeds 600–609, independent initial XY variation ±8 mm. These seeds were not used for training or model selection.

| Instruction | Success |
|---|---|
| Cup first; ten-seed video | 5/10 |
| Utensil first | 5/10 |

Historical-model reference on the same fresh cup-first seeds: 5/10 using PyTorch (a different runtime from the video). Improved validation accuracy does not by itself establish improved held-out task reliability.

The video contains every consecutive seed, including failures, with command, model hash, final outcome and aggregate results. Objects use real simulated contacts. No IK/controller fallback is used by the learned rollout.

## Broader robustness

Separate PyTorch tests, paired seeds 700–709. These are stress tests, not the position-only video. No tuning was performed to these results.

| Profile | Success |
|---|---|
| standard | 6/10 |
| lighting | 4/10 |
| physics | 6/10 |
| size | 7/10 |
| combined | 6/10 |

Ranges and interpretation: [ROBUSTNESS.md](../ROBUSTNESS.md). Size variation is not unfamiliar-shape generalization. Recovery is not implemented.

## OpenVINO and hardware

See checkpoints/openvino-benchmark.json for model parity, weight sizes and timings. Timings were collected while other local jobs were active; they are indicative, not isolated performance measurements. CPU computes in FP32 for both weight formats. Only FP32 weights are used in the final video.

Actual host: Core Ultra 5 125U (Series 1). Intel documentation supports CPU compatibility in principle with Series 2/3; no target-system execution or speedup is claimed. See [compatibility assessment](../COMPATIBILITY.md).

## Limits

Two instruction templates; eighteen-keyframe horizon; strong learned trajectory prior; proxy tableware; plate already positioned; sequential complementary arm actions. This is not a foundation VLA or general multimodal reasoning system. Historical original results are in RESULTS-original.md and must not be combined with this checkpoint.
