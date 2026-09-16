# What broader robustness means

Robustness asks whether the same frozen policy still works when the scene changes.
Training loss alone cannot answer this. Paired tests use identical seed sets and
change one factor at a time, then combine factors. Every failure is retained.

| Factor | Implemented test range | What it probes |
|---|---|---|
| Placement | independent object XY ±8 mm; extra training up to ±12 mm | New starting positions |
| Lighting/background | light RGB intensity 0.55–1.0; table RGB 0.10–0.28 | Visual sensitivity |
| Mass/friction | mass 70–130% of 35 g; sliding friction 0.7–1.3 | Grasp and release dynamics |
| Object size | horizontal dimensions 90–110%; height unchanged | Grasp geometry changes |
| Combined | lighting, background, mass, friction, horizontal size together | Accumulated shift |

Run `python robotics/learning/robustness.py --start 700 --count 10` for 50 episodes,
ten per profile including a paired standard condition. These are deliberately modest
simulation variations. Scaling dimensions is not arbitrary shape generalization.
Results do not establish success with realistic tableware or real robots.

Not implemented as demonstrated capabilities: camera relocation, occlusion, unfamiliar
object shapes, arbitrary instructions, disturbance recovery, retries after a failed
grasp, and multiple place settings. Those would require additional observations,
training examples and closed-loop recovery evaluation. Our fixed eighteen-keyframe
policy currently has no general recovery controller.

The ordinary ten-seed video varies position only. Broader stress results are reported
separately, so a successful position-only demonstration cannot imply all these tests pass.
