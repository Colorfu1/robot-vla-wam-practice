# DPPO local configuration patch

Upstream: https://github.com/irom-princeton/dppo.git

Local checkouts are ignored and are not submodules. Recreate them at these revisions:

| Checkout | Revision | Local changes |
| --- | --- | --- |
| `third_party/dppo/dppo` | `cc7234ad7ff39a8f32de3af903606723a16f0648` | Apply the patch below |
| `third_party/dppo_official` | `cc7234ad7ff39a8f32de3af903606723a16f0648` | None |
| `third_party/dppo_v06` | `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc` | None |

`eval-diffusion-target.patch` preserves the existing 26 evaluation configuration
changes from `model.diffusion.diffusion_eval_ft.DiffusionEval` to
`model.diffusion.diffusion_eval.DiffusionEval`. It does not contain weights or outputs.
This publication preserves existing edits; it does not establish their benchmark validity.

From the repository root, after cloning and checking out the listed revision:

```bash
git -C third_party/dppo/dppo apply --check ../../patches/dppo/eval-diffusion-target.patch
git -C third_party/dppo/dppo apply ../../patches/dppo/eval-diffusion-target.patch
```

The two unmodified versions can be recreated with:

```bash
git clone https://github.com/irom-princeton/dppo.git third_party/dppo_official
git -C third_party/dppo_official checkout --detach cc7234ad7ff39a8f32de3af903606723a16f0648
git clone https://github.com/irom-princeton/dppo.git third_party/dppo_v06
git -C third_party/dppo_v06 checkout --detach dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc
```
