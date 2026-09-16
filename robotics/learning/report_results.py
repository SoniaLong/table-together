"""Build the report directly from complete saved evaluation artifacts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
experiment=read('checkpoints/experiment.json')
video=read('evaluations/ten-seed-final/summary.json')
reverse=read('evaluations/final-utensil-first/summary.json')
original=read('evaluations/final-original-cup/summary.json')
stress=read('evaluations/robustness-improved/summary.json')
assert video['episodes']==reverse['episodes']==10
assert len(stress['results'])==50
digest=(ROOT/'checkpoints/SHA256').read_text().strip()
assert all(x['checkpoint_sha256']==digest for x in [video,reverse,stress])
lines=['# Improved trained-policy results','',f'Frozen checkpoint: `{digest}`.','',
'75 successful demonstrations / 1,350 keyframes, versus the original 43 / 774. '+
'Eight separate validation episodes. Warm-start training: 5,000 iterations, 288.08 seconds on local CPU.',
'Validation-selected iteration 4,200; joint MAE 0.001933 radians versus 0.002598 (25.6% lower).',
'Full validation success: 7/8 versus 6/8 on the same seeds and instruction orders. '+
'This small validation improvement is not a broad reliability guarantee.','',
'## Fresh final evaluation','',
'OpenVINO CPU, seeds 600–609, independent initial XY variation ±8 mm. '+
'These seeds were not used for training or model selection.','',
'| Instruction | Success |','|---|---|',f'| Cup first; ten-seed video | {video["successes"]}/10 |',
f'| Utensil first | {reverse["successes"]}/10 |','',
f'Historical-model reference on the same fresh cup-first seeds: {original["successes"]}/10 '+
'using PyTorch (a different runtime from the video). Improved validation accuracy '+
'does not by itself establish improved held-out task reliability.','',
'The video contains every consecutive seed, including failures, with command, model '+
'hash, final outcome and aggregate results. Objects use real simulated contacts. '+
'No IK/controller fallback is used by the learned rollout.','',
'## Broader robustness','',
'Separate PyTorch tests, paired seeds 700–709. These are stress tests, not the '+
'position-only video. No tuning was performed to these results.','',
'| Profile | Success |','|---|---|']
for profile in ['standard','lighting','physics','size','combined']:
    rows=[r for r in stress['results'] if r['profile']==profile]
    lines.append(f'| {profile} | {sum(r["success"] for r in rows)}/{len(rows)} |')
lines += ['', 'Ranges and interpretation: [ROBUSTNESS.md](../ROBUSTNESS.md). '+
'Size variation is not unfamiliar-shape generalization. Recovery is not implemented.','',
'## OpenVINO and hardware','',
'See checkpoints/openvino-benchmark.json for model parity, weight sizes and timings. '+
'Timings were collected while other local jobs were active; they are indicative, '+
'not isolated performance measurements. CPU computes in FP32 for both weight formats. '+
'Only FP32 weights are used in the final video.','',
'Actual host: Core Ultra 5 125U (Series 1). Intel documentation supports CPU compatibility '+
'in principle with Series 2/3; no target-system execution or speedup is claimed. '+
'See [compatibility assessment](../COMPATIBILITY.md).','',
'## Limits','',
'Two instruction templates; eighteen-keyframe horizon; strong learned trajectory prior; '+
'proxy tableware; plate already positioned; sequential complementary arm actions. '+
'This is not a foundation VLA or general multimodal reasoning system. Historical original '+
'results are in RESULTS-original.md and must not be combined with this checkpoint.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
