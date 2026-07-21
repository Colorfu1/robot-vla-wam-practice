# Third-Party Notices

The root Apache License 2.0 applies to project-owned work. Dependencies, upstream material,
datasets, and trademarks remain governed by their respective owners and licenses.

## LeRobot

This project uses LeRobot as an installed dependency and does not vendor its source code.

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

## Referenced datasets

Datasets are downloaded separately and are not distributed by this repository:

- `lerobot/aloha_mobile_cabinet`:
  https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet — Apache License 2.0.
- `lerobot/libero`:
  https://huggingface.co/datasets/lerobot/libero — Apache License 2.0.

Users must review and follow the current dataset cards, licenses, and upstream attribution or
citation requirements before redistributing data, videos, derived recordings, or trained models.

## Other dependencies

Packages listed in `requirements.txt` and their transitive dependencies are not vendored. Each is
governed by its own license. Product and project names remain trademarks of their respective
owners.
