# Core Ultra Series 2/3 compatibility assessment

Assessment date: 2026-09-16. **Theoretically compatible through CPU inference;
not yet validated on physical Series 2/3 hardware.**

OpenVINO 2026 explicitly lists Intel Core Ultra Series 1, 2 and 3 as supported CPU
hardware, on Windows 11/10 and supported Linux releases:
https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html

MuJoCo supplies x86-64 Windows and Linux binaries and requires AVX. Its classic
renderer uses OpenGL; working graphics drivers/rendering context are required:
https://mujoco.readthedocs.io/en/stable/programming/index.html

Our deployment uses a portable OpenVINO IR graph, FP32 CPU inference, Python, and
MuJoCo CPU physics with offscreen OpenGL rendering. It has no CUDA dependency and
no instruction specific to the current Series 1 CPU. Actual local execution and
export parity checks establish that this graph works through OpenVINO CPU, while
Intel's support list establishes the target CPU family support. Together these
support the compatibility inference; they do not measure target performance.

Use Windows 11 x64 or a supported Ubuntu release, Python 3.10, pinned dependencies,
and working Intel graphics drivers. The CPU path does not require NPU inference.
GPU/NPU deployment is optional and needs separate driver, graph-support, accuracy
and performance checks. Do not infer full-model GPU/NPU compatibility from a tiny
smoke test or from support for the processor family.

## On-device validation

1. Install dependencies from README and fetch pinned robot assets.
2. Run `python robotics/verify_hardware.py`; retain the actual processor name.
3. Run `python robotics/learning/export_openvino.py` for model parity and timing.
4. Run `python robotics/learning/rollout.py --runtime openvino --start 600 --count 10 --label target-hardware`.
5. Record the same ten seeds on that machine and retain all outcomes.

Expected compatibility is not a claim of successful Series 2/3 execution, GPU/NPU
acceleration, equivalent success rates, or any particular speedup. The organizer's
hardware demonstration remains outstanding until those measurements exist.
