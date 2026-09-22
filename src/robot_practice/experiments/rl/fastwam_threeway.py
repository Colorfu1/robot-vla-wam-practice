"""Load genuine intervention rollouts and display independently labelled timelines."""

import json

import numpy as np


def intervention_sources(root, source_hash, trajectory, frames):
    meta = json.loads((root / "metrics.json").read_text())
    if (meta["status"] != "complete" or meta["trajectory"] != trajectory
            or meta["source_metrics_sha256"] != source_hash):
        raise ValueError("Intervention run incomplete or from different trajectory/source evaluation")
    sources, details, excluded = [], {}, {}
    for start, row in sorted(meta["cases"].items(), key=lambda x: int(x[0])):
        policy = row.get("policy_continuation")
        if not row["eligible"] or not policy or not policy["restore_gate"]["passed"]:
            excluded[start] = row.get("policy_skipped_reason", "Policy restore check failed")
            continue
        if (not row["demo_continuation"]["restore_gate"]["passed"]
                or not row["demo_continuation"]["task_success_any_step"]):
            raise ValueError("Inconsistent intervention eligibility")
        path = root / policy["artifact_directory"] / "control_trace.npz"
        with np.load(path) as z:
            data = {key: z[key] for key in ("rgb", "proprio", "actions")}
        if (len(data["rgb"]) != len(data["actions"]) + 1
                or len(data["proprio"]) != len(data["rgb"])):
            raise ValueError("Intervention trace does not contain all post-action observations")
        if any(t < 0 or t + 4 >= len(data["rgb"]) for t in frames):
            raise ValueError("Intervention frames outside recorded future")
        name = f"intervention_{int(start):04d}"
        sources.append((name, data))
        details[name] = {"kind": "intervention", "restored_demo_frame": int(start),
                         "task_success": policy["task_success_any_step"],
                         "time_axis": "Elapsed control steps after reset; NOT original demo indices",
                         "truth": "Actual policy continuation, NOT demonstration continuation"}
    if not sources:
        raise ValueError("No eligible policy intervention traces; cannot build three-way comparison")
    return sources, details, excluded


def threeway_viewer(output, cases, details, excluded):
    payload = json.dumps({"cases": cases, "details": details, "excluded": excluded}).replace(
        "<", "\\u003c")
    (output / "index.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Three-way prediction audit</title>
<style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(3,minmax(360px,1fr));gap:12px}
img{width:100%}pre{white-space:pre-wrap}section{border:1px solid #aaa;padding:8px}</style>
<h1>三组真实输入下的未来预测</h1>
<p>每列：INPUT当前真实输入 / 红字PRED生成的未来 / GT该轨迹自己的真实未来 / COPY复制当前帧。
GT输入组不是自主rollout。干预组是恢复真实示教状态后自主执行，不是将GT图像贴入失败轨迹。</p>
<p>三列独立时间轴，不默认同阶段。每次预测只有未来4步（0.2秒），无未来GT输入，无动作条件。
非4步边界的rollout预测为离线查询，不是当时执行过的策略调用；GT仍来自原有动作序列。
点击图像可查看原尺寸。</p>
<main><section id="rollout"><h2>1 原始失败rollout</h2></section>
<section id="demo"><h2>2 GT示教输入（逐窗口独立）</h2></section>
<section id="intervention"><h2>3 GT状态干预后自主rollout</h2></section></main>
<h3>未通过示教续跑或恢复检查的起点</h3><pre id="excluded"></pre>
<p><a href="all_frames.html">全部预测窗口</a></p>
<script>
const data=PAYLOAD;
document.getElementById('excluded').textContent=JSON.stringify(data.excluded,null,2);
for(const group of ['rollout','demo','intervention']){
 const panel=document.getElementById(group), choose=document.createElement('select');
 const sources=[...new Set(Object.values(data.cases).map(c=>c.source))].filter(
   s=>group==='intervention'?s.startsWith('intervention_'):s===group);
 sources.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;choose.append(o)});
 const note=document.createElement('pre'),prev=document.createElement('button'),next=document.createElement('button');
 prev.textContent='上一输入帧';next.textContent='下一输入帧';
 const slider=document.createElement('input');slider.type='range';slider.min=0;slider.value=0;
 const label=document.createElement('p'),link=document.createElement('a'),image=document.createElement('img');
 link.target='_blank';link.append(image);const metrics=document.createElement('pre');
 panel.append(choose,note,prev,next,slider,label,link,metrics);let rows=[];
 function show(){const [key,c]=rows[Number(slider.value)];
 image.src=key+'/comparison.png';link.href=image.src;
 label.textContent='输入步 '+c.control_step+' → 本轨迹真实未来 '+(c.control_step+1)+'..'+(c.control_step+4);
 metrics.textContent=JSON.stringify({query:c.query_kind,gripper_pred:c.predicted_gripper_commands,
 gripper_reference:c.reference_gripper_commands,action_mae:c.clipped_action_mae_vs_reference,
 image_errors:c.future_frames},null,2);}
 function change(){rows=Object.entries(data.cases).filter(([k,c])=>c.source===choose.value)
 .sort((a,b)=>a[1].control_step-b[1].control_step);slider.max=rows.length-1;slider.value=0;
 note.textContent=JSON.stringify(data.details[choose.value],null,2);show();}
 choose.onchange=change;slider.oninput=show;
 prev.onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show()};
 next.onclick=()=>{slider.value=Math.min(rows.length-1,Number(slider.value)+1);show()};change();
}
</script>'''.replace("PAYLOAD", payload))
