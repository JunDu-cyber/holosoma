"""View the raw human motion (no robot, no retargeting).

Animates the 52 SMPL-H joints stored in an InterMimic/OMOMO ``.pt`` file (or the
preprocessed ``human_joints`` inside a retargeting result ``.npz``) as a stick
figure in viser, with a frame slider. Use this to judge what the *human* actually
does before blaming the retargeting.

Usage (from the holosoma_retargeting package root):

    python examples/view_raw_motion.py demo_data/OMOMO_new/sub3_largebox_003.pt
    python examples/view_raw_motion.py --npz demo_results/n2/robot_only/omomo/sub3_largebox_003.npz
    python examples/view_raw_motion.py demo_data/OMOMO_new/sub3_largebox_003.pt --scale 0.65

``--scale`` applies the same uniform scaling the retargeter uses (robot_height /
subject_height), so you can see the exact reference cloud the solver sees.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import viser


def load_joints(path: str, npz: bool) -> np.ndarray:
    """Return (T, 52, 3) world-space SMPL-H joints."""
    if npz:
        return np.load(path)["human_joints"]
    import torch

    data = torch.load(path, map_location="cpu").detach().numpy()
    return data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)


# SMPL-H joint order (matches DEMO_JOINTS_REGISTRY["smplh"]) and parent indices.
NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]
PARENT = np.array([
    -1, 0, 1, 2, 3, 0, 5, 6, 7,
    0, 9, 10, 11, 12,
    11, 14, 15, 16,
    17, 18, 19, 17, 21, 22, 17, 24, 25, 17, 27, 28, 17, 30, 31,
    11, 33, 34, 35,
    35, 36, 37, 35, 39, 40, 35, 42, 43, 35, 45, 46, 35, 48, 49,
])
BONES = [(p, c) for c, p in enumerate(PARENT) if p >= 0]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    npz_mode = "--npz" in sys.argv
    scale = 1.0
    if "--scale" in sys.argv:
        scale = float(sys.argv[sys.argv.index("--scale") + 1])
    if not args:
        raise SystemExit(__doc__)

    joints = load_joints(args[0], npz_mode) * scale
    T = len(joints)
    print(f"{T} frames, scale={scale:.3f}")

    server = viser.ViserServer(port=8082)
    gui = server.gui.add_folder("raw human motion")
    slider = gui.add_slider("frame", 0, T - 1, step=1, initial_value=0)
    speed = gui.add_slider("playback speed", 0, 4, step=0.25, initial_value=1.0)
    gui.add_button("pause/resume")

    pts = server.scene.add_point_cloud("joints", points=joints[0], point_size=0.02, color=(0.2, 0.4, 1.0))
    lines = server.scene.add_line_segments(
        "skeleton", np.array([[joints[0, p], joints[0, c]] for p, c in BONES]), line_width=2.0
    )
    server.scene.add_grid("ground", width=4, height=4)

    playing = True
    prev_playing = playing

    @slider.on_update
    def _(_: object = slider) -> None:
        f = int(slider.value)
        pts.points = joints[f]
        lines.segments = np.array([[joints[f, p], joints[f, c]] for p, c in BONES])

    @gui.children[-1].on_click
    def _toggle(_: object = gui.children[-1]) -> None:
        nonlocal playing
        playing = not playing

    while True:
        if playing and prev_playing and speed.value > 0:
            nxt = (int(slider.value) + 1) % T
            slider.value = nxt
            pts.points = joints[nxt]
            lines.segments = np.array([[joints[nxt, p], joints[nxt, c]] for p, c in BONES])
            time.sleep(1.0 / (30.0 * float(speed.value)))
        else:
            time.sleep(0.05)
        prev_playing = playing


if __name__ == "__main__":
    main()
