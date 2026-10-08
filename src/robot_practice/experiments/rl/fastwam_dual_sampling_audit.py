"""CPU-only inspection of saved dual-view frames and stride-four video supervision."""

import argparse
import csv
import hashlib
import html
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger(__name__)


def close_onsets(actions):
    close = np.asarray(actions)[:, 6] > 0
    return np.flatnonzero(close & np.r_[True, ~close[:-1]]).tolist()


def frame_coverage(length):
    """Exact possible-window coverage; does not claim windows were sampled in a run."""
    inputs = np.zeros(length, dtype=int)
    futures = np.zeros(length, dtype=int)
    valid = max(0, length - 32)
    inputs[:valid] = 1
    for k in range(1, 9):
        futures[4 * k:4 * k + valid] += 1
    return inputs, futures


def sample_windows(length, onset):
    """Four adjacent real window starts near command onset, with distinct phases."""
    windows = []
    for phase in range(4):
        start = max(0, onset - 4) + phase
        if start < length - 32:
            windows.append({"start": start, "phase_mod4": start % 4,
                            "frames": (start + np.arange(9) * 4).tolist()})
    return windows


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def save_sheet(directory, frames, onset, images, begin, actions, name):
    canvas = Image.new("RGB", (448 * 3, 254 * ((len(frames) + 2) // 3)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        x, y = (i % 3) * 448, (i // 3) * 254
        draw.text((x + 4, y + 3),
                  f"frame={frame} onset{frame-onset:+d} grip_cmd={actions[frame, 6]:+.3f}",
                  fill="black")
        canvas.paste(Image.fromarray(images[begin + frame].transpose(1, 2, 0)), (x, y + 30))
    canvas.save(directory / name)


def export_event(root, trajectory, onset, images, states, actions, begin, before, after):
    length = len(actions)
    windows = sample_windows(length, onset)
    lo, hi = max(0, onset - before), min(length - 1, onset + after)
    selected = [f for w in windows for f in w["frames"]]
    if selected:
        lo, hi = min(lo, min(selected)), max(hi, max(selected))
    frames = list(range(lo, hi + 1))
    directory = root / f"trajectory_{trajectory}" / f"close_command_{onset:04d}"
    directory.mkdir(parents=True)
    inputs, futures = frame_coverage(length)
    rows = []
    for frame in frames:
        rgb = images[begin + frame]
        Image.fromarray(rgb.transpose(1, 2, 0)).save(directory / f"frame_{frame:04d}.png")
        row = {"frame": frame, "seconds_from_close_command": (frame - onset) / 20,
               "grip_command_normalized": float(actions[frame, 6]),
               "eef_z_normalized": float(states[begin + frame, 2]),
               "finger1_q_normalized": float(states[begin + frame, 7]),
               "finger2_q_normalized": float(states[begin + frame, 8]),
               "possible_input_windows": int(inputs[frame]),
               "possible_future_supervision_windows": int(futures[frame])}
        if frame > 0:
            diff = np.abs(rgb.astype(float) - images[begin + frame - 1].astype(float))
            row.update(agentview_adjacent_mae=float(diff[:, :, :224].mean()),
                       wrist_adjacent_mae=float(diff[:, :, 224:].mean()))
        else:
            row.update(agentview_adjacent_mae=None, wrist_adjacent_mae=None)
        rows.append(row)
    with (directory / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sections = []
    for offset in range(0, len(frames), 12):
        name = f"continuous_{offset:03d}.png"
        save_sheet(directory, frames[offset:offset + 12], onset, images, begin, actions, name)
        sections.append(f'<a href="{name}"><img width="100%" src="{name}"></a>')
    samples = []
    for window in windows:
        name = f'window_start_{window["start"]:04d}.png'
        save_sheet(directory, window["frames"], onset, images, begin, actions, name)
        samples.append(f'<h3>实际训练窗口起点 {window["start"]}，'
                       f'相位 {window["phase_mod4"]}：{window["frames"]}</h3>'
                       f'<a href="{name}"><img width="100%" src="{name}"></a>')
    payload = json.dumps(rows)
    page = f'''<!doctype html><meta charset="utf-8">
<title>Trajectory {trajectory}, close command {onset}</title>
<style>body{{font:16px sans-serif;max-width:1400px;margin:24px auto;padding:12px}}
button,select{{padding:8px}} #viewer{{image-rendering:auto;max-width:100%}}</style>
<h1>轨迹 {trajectory} / 闭合指令起点 {onset}</h1>
<p>所有画面都是保存的训练数据，不是模型预测，也不是新rollout。
左：agentview；右：腕相机。PNG保留各相机原生224×224。
画面 frame=t 在 action[t] 执行之前。闭合指令不代表接触或抓稳。</p>
<p>先逐帧确认：何时接触、物体何时随夹爪移动、腕相机是否遮挡。
下面是原始连续帧以及4个相邻起点的真实9帧训练窗口。</p>
<button onclick="move(-1)">上一原始帧</button>
<button onclick="move(1)">下一原始帧</button>
<button onclick="play(1)">连续20Hz（慢放4倍）</button>
<button onclick="play(4)">每4帧采样（慢放4倍）</button>
<button onclick="stop()">暂停</button>
<p>采样相位由播放起点决定；拖动一帧再播放可比较另一相位。</p>
<input id="slider" type="range" min="0" max="{len(rows)-1}" value="0"
 style="width:95%" oninput="stop();index=Number(this.value);show()">
<pre id="label"></pre><img id="viewer" width="896">
<p><a href="frames.csv">逐帧CSV（关节和末端数值是归一化量）</a></p>
<h2>逐帧原图：无跳帧</h2>{''.join(sections)}
<h2>视频监督：每个窗口9帧，间隔4控制步</h2>
{''.join(samples) if samples else '<p>附近没有完整32步的合法训练窗口。</p>'}
<script>
const rows={payload};let index=0,timer=null;
function stop(){{clearInterval(timer);timer=null}}
function show(){{let r=rows[index];document.getElementById('slider').value=index;
document.getElementById('viewer').src='frame_'+String(r.frame).padStart(4,'0')+'.png';
document.getElementById('label').textContent=JSON.stringify(r,null,2)}}
function move(n){{stop();index=Math.max(0,Math.min(rows.length-1,index+n));show()}}
function play(stride){{stop();timer=setInterval(()=>{{if(index+stride>=rows.length){{stop();return}}
index+=stride;show()}},200*stride)}}show();
</script>'''
    (directory / "index.html").write_text(page)
    return {"trajectory": trajectory, "close_command_frame": onset, "range": [lo, hi],
            "video_windows": windows, "page": str(directory.relative_to(root) / "index.html"),
            "physical_contact_frame": None, "object_lift_frame": None,
            "wrist_occlusion_review": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/derived/square_dual224_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-ids", default="0,10,20,30,40,50,60,70")
    parser.add_argument("--before", type=int, default=20)
    parser.add_argument("--after", type=int, default=44)
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    if (manifest["status"] != "complete" or protocol["video_stride"] != 4 or
            protocol["video_frames"] != 9 or protocol["action_horizon"] != 32 or
            protocol["control_hz"] != 20 or protocol["image_size"] != [224, 448]):
        raise ValueError("Expected complete dual224, horizon32, stride4, 20Hz dataset")
    lengths = np.load(args.data / "traj_lengths.npy", allow_pickle=False)
    images, states, actions = [np.load(args.data / f"{key}.npy", mmap_mode="r",
                                     allow_pickle=False) for key in ("images", "states", "actions")]
    n = int(lengths.sum())
    if (images.shape != (n, 3, 224, 448) or images.dtype != np.uint8 or
            actions.shape != (n, 7) or states.shape != (n, 9)):
        raise ValueError("Unexpected dataset shapes/dtype")
    ids = [int(x) for x in args.trajectory_ids.split(",")]
    if (not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(lengths)
            or args.before < 0 or args.after < 0):
        parser.error("Invalid trajectories or frame range")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "protocol": protocol, "events": [], "coverage": [],
              "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_data": str(args.data), "selected_trajectories": ids,
              "limitations": ["Coverage counts possible windows, not actual training draws.",
                              "Close commands do not establish grasp or object lift.",
                              "No inference, rendering, or camera ablation is performed."]}
    write_json(args.output / "metrics.json", result)
    try:
        starts = np.r_[0, np.cumsum(lengths)]
        for tid, length in enumerate(lengths):
            inputs, futures = frame_coverage(int(length))
            result["coverage"].append({"trajectory": tid, "frames": int(length),
                "possible_window_starts": int(inputs.sum()),
                "frames_never_input": int((inputs == 0).sum()),
                "frames_never_future_target": int((futures == 0).sum()),
                "frames_never_used_in_video": int(((inputs + futures) == 0).sum())})
        events = [(tid, onset) for tid in ids for onset in close_onsets(
            actions[int(starts[tid]):int(starts[tid + 1])])]
        result["requested_events"] = len(events)
        for number, (tid, onset) in enumerate(events, 1):
            begin, end = int(starts[tid]), int(starts[tid + 1])
            result["events"].append(export_event(args.output, tid, onset, images, states,
                                    actions[begin:end], begin, args.before, args.after))
            write_json(args.output / "metrics.json", result)
            LOG.info("STATUS phase=data_audit completed=%d total=%d trajectory=%d onset=%d",
                     number, len(events), tid, onset)
        links = ''.join(f'<li><a href="{html.escape(e["page"])}">trajectory={e["trajectory"]} '
                        f'闭合指令起点={e["close_command_frame"]}</a></li>' for e in result["events"])
        (args.output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><h1>双相机训练数据采样检查</h1>'
            '<p>连续20Hz真值图像 vs stride=4视频监督；只检查数据，不评测策略。</p>'
            '<p>每个页面左侧为外部相机、右侧为腕相机；对照连续帧判断接触与抬起，'
            '再看4种采样相位是否遗漏关键画面。相机优劣需要后续受控实验。</p><ul>'
            + links + '</ul>')
        result["status"] = "complete"
        LOG.info("DONE phase=data_audit completed=%d total=%d", len(events), len(events))
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
