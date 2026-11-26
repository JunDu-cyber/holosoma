"""Decisive side-correspondence test: render the solved N2 pose through the MuJoCo
XML (the solver's ground-truth model) with side-colored markers:

  human landmarks: blue = left side, red = right side, gray = center
  robot links:     green = left-named link, orange = right-named link

If green spheres cluster with blue dots, the retargeting tracks the correct sides
and any perceived diagonal failure comes from the *display* URDF or viewing
convention. If green clusters with red, the solver itself has a side swap.

Usage (from the holosoma_retargeting package root; needs a GL backend, tries EGL):

    python examples/render_side_check.py demo_results/n2/robot_only/omomo/sub10_largebox_049.npz

Renders a standing and a deep-crouch frame from front-3/4 and side views to
/tmp/opencode/renders/, and prints same-side vs cross-side tracking errors.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np

PKG = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(PKG))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, JOINTS_MAPPINGS

npz_path = Path(sys.argv[1] if len(sys.argv) > 1 else PKG / "holosoma_retargeting/demo_results/n2/robot_only/omomo/sub10_largebox_049.npz")
out_dir = Path("/tmp/opencode/renders")
out_dir.mkdir(parents=True, exist_ok=True)

res = np.load(str(npz_path))
qpos, human = res["qpos"], res["human_joints"]
T = len(qpos)

mapping = JOINTS_MAPPINGS[("smplh", "n2")]
demo = list(DEMO_JOINTS_REGISTRY["smplh"])

m = mj.MjModel.from_xml_path(str(PKG / "holosoma_retargeting/models/n2/n2_18dof.xml"))
d = mj.MjData(m)


def marker(scene, pos, rgba, radius):
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom, mj.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=float), np.eye(3).flatten(), np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


# pick two frames: deepest crouch (min human pelvis z) and a tall/standing one
pelv = human[:, demo.index("Pelvis"), 2]
frames = {"stand": int(np.argmax(pelv)), "crouch": int(np.argmin(pelv))}

renderer = mj.Renderer(m, height=480, width=640)

for label, f in frames.items():
    d.qpos[:] = qpos[f]
    mj.mj_forward(m, d)
    pelvis = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base_link")]

    for view, (azimuth, elevation) in {
        "front34": (135.0, -18.0),
        "side": (90.0, -12.0),
    }.items():
        cam = mj.MjvCamera()
        cam.lookat = pelvis + np.array([0, 0, -0.05])
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, 2.6
        renderer.update_scene(d, cam)
        scene = renderer._scene  # inject markers into the renderer's own scene

        for name, body in mapping.items():
            hp = human[f, demo.index(name)]
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body)
            rp = d.xpos[bid]
            if name.startswith("L"):
                marker(scene, hp, (0.2, 0.45, 1.0, 1.0), 0.030)  # human left: blue
                marker(scene, rp, (0.1, 0.9, 0.2, 1.0), 0.022)   # robot left: green
            elif name.startswith("R"):
                marker(scene, hp, (1.0, 0.25, 0.25, 1.0), 0.030)  # human right: red
                marker(scene, rp, (1.0, 0.55, 0.1, 1.0), 0.022)   # robot right: orange
            else:
                marker(scene, hp, (0.5, 0.5, 0.5, 1.0), 0.028)
                marker(scene, rp, (0.5, 0.5, 0.5, 1.0), 0.020)

        img = renderer.render()
        out = out_dir / f"{npz_path.stem}_{label}_{view}.png"
        __import__("cv2").imwrite(str(out), img[..., ::-1])
        print(f"saved {out}")

# also report the numerical side-correspondence, frame-independent:
# correlation between (robot left link - human left landmark) vs cross-side distances
left_names = [n for n in mapping if n.startswith("L")]
right_names = [n for n in mapping if n.startswith("R")]
same, cross = [], []
for f in (frames["stand"], frames["crouch"]):
    d.qpos[:] = qpos[f]
    mj.mj_forward(m, d)
    for ln, rn in zip(left_names, right_names):
        lb = mapping[ln]
        rb = mapping[rn]
        pl = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, lb)]
        pr = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb)]
        hl = human[f, demo.index(ln)]
        hr = human[f, demo.index(rn)]
        same += [np.linalg.norm(pl - hl), np.linalg.norm(pr - hr)]
        cross += [np.linalg.norm(pl - hr), np.linalg.norm(pr - hl)]
print(f"\nSAME-side tracking error (green-vs-blue):  mean {np.mean(same):.4f} m")
print(f"CROSS-side error (green-vs-red):          mean {np.mean(cross):.4f} m")
print("If same << cross, the solver tracks the correct sides.")
