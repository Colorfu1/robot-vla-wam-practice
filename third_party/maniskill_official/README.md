# Bundled ManiSkill Diffusion Policy baseline

This directory vendors the upstream ManiSkill Diffusion Policy baseline needed by
the official StackCube RGB golden-anchor experiment. It is kept in the upstream
checkout layout so the runner can execute the official `train_rgbd.py` without
network access or an online git clone.

- Upstream repository: https://github.com/haosulab/ManiSkill
- Source revision: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`
- Upstream baseline path: `examples/baselines/diffusion_policy`
- License: Apache-2.0; see [LICENSE](LICENSE)

The bundled files are source dependencies, not generated experiment outputs.

The bundled `diffusion_policy/evaluate.py` contains one compatibility adapter for
the `info["episode"]` / `_episode` schema emitted by Gymnasium 1.x with the
installed ManiSkill CPU wrapper. It preserves the official episode metrics and
also accepts the older `info["final_info"]` schema.
