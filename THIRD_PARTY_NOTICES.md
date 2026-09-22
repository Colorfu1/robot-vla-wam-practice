# Third-Party Notices

The root Apache License 2.0 applies to project-owned work. Dependencies, upstream material,
datasets, and trademarks remain governed by their respective owners and licenses.

## LeRobot

This project uses LeRobot as an installed dependency rather than bundling a full source
checkout. A local loader patch is distributed separately and is described below.

- Project: https://github.com/huggingface/lerobot
- License: Apache License 2.0
- Copyright: the LeRobot contributors

## gym-aloha and the original ALOHA project

`src/robot_practice/aloha_kinematics.py` uses nominal ViperX 300S geometry and puppet-gripper
calibration values adapted from `gym-aloha`. The FK implementation itself is local project code.
`tests/test_aloha_kinematics.py` contains numerical golden values produced from the corresponding
MJCF model; it does not copy an upstream source file.

- Project: https://github.com/huggingface/gym-aloha
- License: Apache License 2.0
- Copyright: 2024 The Hugging Face team

The `gym-aloha` distribution states that it is derived from the original ALOHA project and
retains the following MIT notice:

> MIT License
>
> Copyright (c) 2023 Tony Z. Zhao
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Bundled ManiSkill baseline and derived geometry

- Project: https://github.com/haosulab/ManiSkill
- Bundled revision: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`
- Scope: `third_party/maniskill_official/examples/baselines/diffusion_policy/`.
- License: Apache License 2.0; the full bundled copy is
  [third_party/maniskill_official/LICENSE](third_party/maniskill_official/LICENSE).
- Provenance and local changes:
  [third_party/maniskill_official/README.md](third_party/maniskill_official/README.md).
  The bundled `diffusion_policy/evaluate.py` was modified for Gymnasium episode-info
  compatibility; it carries a modification notice in the file.

The baseline README attributes the implementation to
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy), whose upstream
MIT notice is also retained in
[Diffusion-Policy-MIT.txt](third_party/licenses/Diffusion-Policy-MIT.txt)
(Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab).
ManiSkill's Apache license does not remove the MIT notice for that inherited material.
The Diffusion Policy notice was checked against upstream on 2026-09-23; this attribution
does not establish an exact historical revision for every inherited baseline component.

`src/robot_practice/experiments/maniskill/panda_kinematics.py` identifies fixed joint
geometry copied from ManiSkill's `panda_v2.urdf`. Its local numerical implementation
does not grant additional rights to robot meshes, textures, or other upstream assets;
check the corresponding asset distribution separately before redistributing those assets.

## DPPO patches

- Project: https://github.com/irom-princeton/dppo
- Patch base revision: `cc7234ad7ff39a8f32de3af903606723a16f0648`
- Copyright (c) 2024 Intelligent Robot Motion Lab
- License: MIT; full notice: [DPPO-MIT.txt](third_party/licenses/DPPO-MIT.txt).
- Scope: `docker/training/vendor-patches/dppo-eval-target.patch` and
  `third_party/patches/dppo/eval-diffusion-target.patch`.

These patches record local changes to the upstream evaluation configuration target.
Their upstream context and any redistributed patched DPPO source retain the MIT notice.
The local upstream checkouts are ignored; their pinned revisions and patch instructions
are recorded in [third_party/patches/dppo/README.md](third_party/patches/dppo/README.md).

## FastWAM and Light-WAM patches

| Upstream | Patch base revision | Retained license |
| --- | --- | --- |
| [FastWAM](https://github.com/yuantianyuan01/FastWAM) | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | [MIT](third_party/licenses/FastWAM-MIT.txt), Copyright (c) 2026 The FastWAM Authors |
| [Light-WAM](https://github.com/L1ziang/Light-WAM) | `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae` | [MIT](third_party/licenses/Light-WAM-MIT.txt), Copyright (c) 2026 The FastWAM Authors and The Light-WAM Authors |

Scope: `fastwam-*.patch` and `lightwam-trainer-staged-stop.patch` under
`docker/training/vendor-patches/`. Local modifications and application status are
described in its [README](docker/training/vendor-patches/README.md).
Full notices above were copied from the pinned upstream Git revisions.
Both licenses expressly preserve the original licenses of third-party files with
their own headers. Their MIT grants must not be extended to all nested dependencies
or downloaded model weights by assumption.

## LeRobot loader patch

`docker/training/vendor-patches/lerobot-local-wan-loader.patch` targets LeRobot
revision `30da8e687a6dfc617fcd94afc367ac7071c376ce`, as pinned by the W0 Dockerfile.
It retains the upstream Apache-2.0 terms and the Hugging Face attribution above.
The patch and the vendor-patches README identify the changes; any redistributed
patched files must also carry modification notices and retain applicable upstream notices.

## Referenced datasets

Datasets are downloaded separately and are not distributed by this repository:

- `lerobot/aloha_mobile_cabinet`:
  https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet — Apache License 2.0.
- `lerobot/libero`:
  https://huggingface.co/datasets/lerobot/libero — Apache License 2.0.

Users must review and follow the current dataset cards, licenses, and upstream attribution or
citation requirements before redistributing data, videos, derived recordings, or trained models.

## Other dependencies

Except for the explicitly bundled source and patch material listed above, package dependencies
are installed or downloaded separately. Requirements and lockfiles do not relicense them.
Redistributing a Docker image also redistributes the dependencies included in that image:
retain their applicable license and notice files, and review any binary or asset restrictions.
Product and project names remain trademarks of their respective owners.

## Review scope

The 2026-09-23 review covers the root license, the directly bundled ManiSkill baseline,
the recorded upstream patches, and the notices listed here. It is not a complete audit of
all transitive dependencies, base images, robot assets, dataset samples, model checkpoints,
or illustrative media. Those materials require their own version-specific provenance and
license checks before reuse or redistribution; the root Apache license is not a substitute.
