"""View the raw human motion (no robot, no retargeting).

Animates the 52 SMPL-H joints stored in an InterMimic/OMOMO ``.pt`` file (or the
preprocessed ``human_joints`` inside a retargeting result ``.npz``) as a stick
figure in viser, with a frame slider. Use this to judge what the *human* actually
does before blaming the retargeting.

Usage (from the holosoma_retargeting package root):

    python examples/view_raw_motion.py demo_data/OMOMO_new/sub3_largebox_003.pt
    python examples/view_raw_motion.py --npz demo_results/n2/robot_only/omomo/sub3_largebox_003.npz
    python examples/view_raw_motion.py demo_data/OMOMO_new/sub3_largebox_003.pt --scale 0.63

``--scale`` applies the same uniform scaling the retargeter uses (robot_height /
subject_height), so you can see the exact reference cloud the solver sees.

Scene and GUI calls copy the repo's own proven viser usage verbatim
(``interaction_mesh_retargeter.draw_keypoints`` for the joint spheres,
``viser_player.py`` for the GUI): joints are drawn with
``add_batched_meshes_simple`` and bones with the ``points``/``colors`` form of
``add_line_segments``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

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
BONE_COLOR = np.array([0.85, 0.85, 0.85])
JOINT_COLOR = (51, 102, 255)  # int rgb, the convention draw_keypoints uses


def load_joints(path: str, npz: bool) -> np.ndarray:
    """Return (T, 52, 3) world-space SMPL-H joints."""
    if npz:
        return np.load(path)["human_joints"]
    import torch

    data = torch.load(path, map_location="cpu").detach().numpy()
    return data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)


def bone_points(frame: np.ndarray, paired: bool) -> np.ndarray:
    """Bone endpoints: ``(N, 2, 3)`` when paired else the flat ``(2N, 3)`` form."""
    segs = np.array([[frame[p], frame[c]] for p, c in BONES])
    return segs if paired else segs.reshape(-1, 3)


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
    with server.gui.add_folder("raw human motion"):
        slider = server.gui.add_slider("frame", 0, T - 1, step=1, initial_value=0)
        speed = server.gui.add_slider("playback speed", 0, 4, step=0.25, initial_value=1.0)
        pause_btn = server.gui.add_button("pause/resume")

    server.scene.add_grid("/ground", width=4, height=4, position=(0.0, 0.0, 0.0))
    sphere = trimesh.primitives.Sphere(radius=0.02)
    handles: list = []

    # viser releases disagree on add_line_segments' points shape: newer ones want
    # (N, 2, 3), older ones (as in interaction_mesh_retargeter) want flat (2N, 3).
    # Probe once and remember which form this install accepts.
    paired_segments: bool | None = None

    def add_skeleton(frame: int):
        nonlocal paired_segments
        if paired_segments is None:
            try:
                handle = server.scene.add_line_segments(
                    "/skeleton",
                    points=bone_points(joints[frame], paired=True),
                    colors=np.tile(BONE_COLOR, (len(BONES), 2, 1)),
                    line_width=0.01,
                )
                paired_segments = True
                return handle
            except ValueError:
                paired_segments = False
        n = 2 * len(BONES)
        if paired_segments:
            colors = np.tile(BONE_COLOR, (len(BONES), 2, 1))
        else:
            colors = np.tile(BONE_COLOR, (n, 1))
        return server.scene.add_line_segments(
            "/skeleton",
            points=bone_points(joints[frame], paired=paired_segments),
            colors=colors,
            line_width=0.01,
        )

    def show(frame: int) -> None:
        # Recreate the two handles each frame, exactly like the retargeter's
        # debug visualization does; no version-sensitive property assignment.
        for handle in handles:
            handle.remove()
        handles.clear()
        handles.append(
            server.scene.add_batched_meshes_simple(
                "/joints",
                vertices=sphere.vertices,
                faces=sphere.faces,
                batched_positions=joints[frame],
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (joints[frame].shape[0], 1)),
                batched_colors=JOINT_COLOR,
                opacity=1.0,
            )
        )
        handles.append(add_skeleton(frame))

    playing = [True]

    @slider.on_update
    def _on_slider(_: object = slider) -> None:
        show(int(slider.value))

    @pause_btn.on_click
    def _on_pause(_: object = pause_btn) -> None:
        playing[0] = not playing[0]

    show(0)
    while True:
        if playing[0] and float(speed.value) > 0:
            nxt = (int(slider.value) + 1) % T
            slider.value = nxt
            show(nxt)
            time.sleep(1.0 / (30.0 * float(speed.value)))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
