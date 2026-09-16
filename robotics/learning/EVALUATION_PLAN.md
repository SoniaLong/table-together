# Improved-training experiment

Predeclared evaluation protocol, before final candidate testing:

- Preserve the original checkpoint and its 14/20 PyTorch result as historical evidence.
- Add expert training seeds 400–415 at ±12 mm XY variation, both orders.
- Original training seeds 100–123 and validation 200–203 remain separate.
- Candidate v2 uses separate output heads per ordinal keyframe, brightness augmentation
  (0.85–1.15) and 0.001-radian joint-input noise.
- Scratch candidate did not improve supervised validation. Warm-start candidate
  initializes shared layers from the original trained checkpoint and replicates its
  residual output head, then trains with Adam learning rate 0.0002 for 5,000 steps.
- Select weights by validation action error. Compare complete episodes on validation
  seeds 200–203 in both instruction orders before deployment.
- Final untouched test/video seeds: 600–609. Do not tune to their outcomes or omit failures.
- Broader stress report uses paired profiles at seeds 700–709. Results are separate
  from the position-only video. The original model's 700–702 probe is development evidence.

If validation rollouts regress, keep the original deployment and report the unsuccessful
training experiment. Model replacement must be justified by measured validation behavior.
