"""Extra camera and contact evidence, never passed to the policy."""

import json

import numpy as np


def select_wrist(cameras):
    candidates = [name for name in cameras if "eye_in_hand" in name.lower()]
    return candidates[0] if len(candidates) == 1 else None


class SceneViews:
    def __init__(self, env, directory):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        self.env, self.directory = env, directory
        self.sim = env.env.sim
        self.cameras = list(self.sim.model.camera_names)
        self.wrist = select_wrist(self.cameras)
        # robot_env applies its convention, then robomimic flips image observations.
        self.flip = -IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.rows = []
        self.images = directory / "views"
        self.images.mkdir()

    def render(self, camera, size):
        return self.sim.render(width=size, height=size, camera_name=camera,
                               depth=False)[::self.flip].copy()

    def append(self, obs):
        from PIL import Image, ImageDraw

        from robot_practice.experiments.rl.grasp_probe import snapshot

        step = len(self.rows)
        evidence = snapshot(self.env.env)
        low = np.clip(obs["rgb"].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        check = self.render("agentview", 96)
        high = self.render("agentview", 224)
        wrist = self.render(self.wrist, 224) if self.wrist else np.zeros_like(high)
        # Lossless individual native images plus a consistently labelled comparison.
        Image.fromarray(low).save(self.images / f"{step:04d}_policy96.png")
        Image.fromarray(check).save(self.images / f"{step:04d}_fresh96.png")
        Image.fromarray(high).save(self.images / f"{step:04d}_agent224.png")
        if self.wrist:
            Image.fromarray(wrist).save(self.images / f"{step:04d}_wrist224.png")
        panels = [Image.fromarray(low).resize((224, 224), Image.Resampling.NEAREST),
                  Image.fromarray(check).resize((224, 224), Image.Resampling.NEAREST),
                  Image.fromarray(high), Image.fromarray(wrist)]
        strip = Image.new("RGB", (896, 252), "white")
        draw = ImageDraw.Draw(strip)
        labels = ["Policy 96 (nearest zoom)", "Fresh native 96 (zoom)", "Native agentview 224",
                  self.wrist or "Wrist unavailable / ambiguous"]
        for i, (panel, label) in enumerate(zip(panels, labels)):
            strip.paste(panel, (224 * i, 28))
            draw.text((224 * i + 3, 5), label, fill="black")
        strip.save(self.images / f"{step:04d}_strip.png")
        nut = evidence["nuts"].get("SquareNut", {})
        height = nut.get("position", [None, None, None])[2]
        initial = self.rows[0]["square_nut_height_m"] if self.rows else height
        evidence.update(
            control_state_index=step, square_nut_height_m=height,
            square_nut_delta_height_m=None if height is None else height - initial,
            fresh96_vs_policy_mae=float(np.abs(check.astype(float) - low).mean()),
            task_success=bool(self.env.is_success()["task"]),
        )
        self.rows.append(evidence)
        with (self.directory / "view_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(evidence) + "\n")

    def finish(self):
        summary = {
            "camera_names": self.cameras, "wrist_camera": self.wrist,
            "wrist_available": self.wrist is not None, "native_resolution": [224, 224],
            "frames": len(self.rows), "policy_input_unchanged": True,
            "fresh96_vs_policy_mae_max": max(r["fresh96_vs_policy_mae"] for r in self.rows),
            "limitations": ["Endpoint contacts only, not all physics substeps.",
                            "Contact or height alone does not establish a successful grasp.",
                            "Missing/ambiguous wrist camera is reported, never substituted.",
                            "Rendering noninterference requires the separate prefix replay gate."],
        }
        (self.directory / "views.json").write_text(json.dumps(summary, indent=2))
        (self.directory / "views.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Same-state camera diagnostic</title>
<h2>Policy 96 / fresh external 96 / fresh external 224 / fresh wrist 224</h2>
<p>Diagnostic cameras are NOT policy inputs. State t is before action t and after action t-1.
Compare fresh 96 vs fresh 224 to isolate resolution. Policy observation may be cached.
Contacts are control-step endpoints; height/contact does not prove grasp success.</p>
<input id="step" type="range" min="0" value="0"><span id="label"></span><br>
<img id="strip" width="1008" style="max-width:100%"><pre id="details"></pre>
<script>
fetch('view_steps.jsonl').then(r=>r.text()).then(t=>{
 const rows=t.trim().split('\\n').map(JSON.parse);
 const slider=document.getElementById('step'); slider.max=rows.length-1;
 function show(){const n=Number(slider.value); document.getElementById('label').textContent=
 ' state '+n+' / '+(rows.length-1)+' ; '+(n/20).toFixed(2)+' s';
 document.getElementById('strip').src='views/'+String(n).padStart(4,'0')+'_strip.png';
 const r=rows[n];document.getElementById('details').textContent=JSON.stringify({
 task_success:r.task_success, square_nut_height_m:r.square_nut_height_m,
 square_nut_delta_height_m:r.square_nut_delta_height_m, gripper_qpos:r.gripper_qpos,
 contacts:r.contacts_endpoint_only, fresh96_vs_policy_mae:r.fresh96_vs_policy_mae},null,2);}
 slider.oninput=show;show();
}).catch(e=>document.getElementById('details').textContent='Serve over HTTP: '+e);
</script>''')
        return summary
