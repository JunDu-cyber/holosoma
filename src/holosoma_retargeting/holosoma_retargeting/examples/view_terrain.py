"""View a reconstructed climbing terrain (terrain.json) with the human motion.

Draws, in viser, exactly what the retargeter sees:

- every fitted box (``terrain.json``) at robot scale, i.e. scaled by the same
  smpl_scale the climbing loader applies (robot_height / 1.78 by default);
- the flat floor grid;
- the human motion (the task's ``*_joint_positions.npy``) with the same
  y-up->z-up + smpl_scale transform the solver applies;
- the footfalls the terrain was fit FROM: toe positions while the toe is
  planted (low speed), colored by foot.

Use this to sanity-check the TERRA fit before trusting a retarget: treads
should sit under the planted feet, tops should match contact heights.

Usage (from the holosoma_retargeting package root):

    python examples/view_terrain.py demo_data/climb_lafan/bones_up_start_r001
    python examples/view_terrain.py demo_data/climb_lafan/bones_up_start_r001 --port 8084

Viser calls mirror ``examples/view_raw_motion.py`` (the repo's proven,
version-proof idioms).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY
from holosoma_retargeting.src.terrain_reconstruction import TerrainSpec

# lafan skeleton parenting (indices follow DEMO_JOINTS_REGISTRY["lafan"]).
LAFAN_PARENT = {
    "Hips": -1,
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
}
BONE_COLOR = np.array([0.85, 0.85, 0.85])
JOINT_COLOR = (51, 102, 255)


def y_up_to_z_up(joints: np.ndarray) -> np.ndarray:
    """Same axis swap the climbing loader applies to lafan sources.

    Mirrors ``transform_y_up_to_z_up`` in src/utils.py exactly: y (up)
    becomes z (up), z (forward) becomes y, x stays - a plain swap, no flip.
    """
    out = joints.copy()
    out[..., [1, 2]] = joints[..., [2, 1]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_dir", type=Path, help="climbing task dir containing terrain.json")
    parser.add_argument("--data_format", default="lafan", choices=["lafan", "mocap"])
    parser.add_argument("--robot_height", type=float, default=1.219, help="robot height for the world scale")
    parser.add_argument("--human_height", type=float, default=1.78, help="subject height of the motion source")
    parser.add_argument("--human_scale", action="store_true", help="show the 1:1 human world instead of the robot world")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    with open(args.task_dir / "terrain.json") as f:
        spec = TerrainSpec.from_dict(json.load(f)["terrain"])

    npy_files = sorted(args.task_dir.glob("*.npy"))
    joints = np.load(npy_files[0]) if npy_files else None
    demo = list(DEMO_JOINTS_REGISTRY[args.data_format])
    if joints is not None and joints.shape[1] != len(demo):
        raise SystemExit(f"motion npy has {joints.shape[1]} joints, {args.data_format} expects {len(demo)}")
    if args.data_format == "lafan":
        joints = y_up_to_z_up(joints)

    scale = args.robot_height / args.human_height
    if args.human_scale:
        scale = 1.0
    world = "human (1:1)" if args.human_scale else f"robot (x{scale:.3f})"
    print(f"{len(spec.boxes)} boxes, {0 if joints is None else len(joints)} frames, world: {world}")

    server = viser.ViserServer(port=args.port)
    with server.gui.add_folder("terrain + motion"):
        if joints is not None:
            slider = server.gui.add_slider("frame", 0, len(joints) - 1, step=1, initial_value=0)
            speed = server.gui.add_slider("playback speed", 0, 4, step=0.25, initial_value=1.0)
            pause_btn = server.gui.add_button("pause/resume")

    server.scene.add_grid("/ground", width=8, height=8, position=(0.0, 0.0, 0.0))

    # terrain boxes, drawn exactly as the solver's collision geoms sit: bake
    # pos+quat+scale into one merged world-space mesh, one add_mesh_simple
    # (the repo's proven single-mesh idiom, same as draw_mesh_from_geom).
    parts = []
    for b in spec.boxes:
        w, x, y, z = b.quat
        R = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]
        local = trimesh.primitives.Box(extents=np.array(b.size) * 2)
        verts = np.asarray(local.vertices) * scale @ R.T + np.array(b.pos) * scale
        faces = np.asarray(local.faces) + sum(len(p["vertices"]) for p in parts)
        parts.append({"vertices": verts, "faces": faces})
    server.scene.add_mesh_simple(
        "/terrain",
        vertices=np.concatenate([p["vertices"] for p in parts]).astype(np.float32),
        faces=np.concatenate([p["faces"] for p in parts]).astype(np.int32),
        position=(0.0, 0.0, 0.0),
        color=(120, 120, 130),
        opacity=0.85,
    )

    # footfalls: planted-toe positions, the evidence TERRA fit the terrain from
    if joints is not None:
        toe_names = ("LeftToeBase", "RightToeBase")
        toe_idx = [demo.index(n) for n in toe_names]
        vel = np.linalg.norm(np.diff(joints[:, toe_idx], axis=0), axis=2)
        planted = vel < 0.01  # toe "still" between frame t and t+1 -> planted at t
        for k, color in zip(range(2), [(255, 60, 60), (60, 120, 255)]):
            pts = joints[:-1][planted[:, k]][:, toe_idx[k]] * scale
            if len(pts):
                dot = trimesh.primitives.Sphere(radius=0.012)
                server.scene.add_batched_meshes_simple(
                    f"/footfall_{'LR'[k]}",
                    vertices=dot.vertices,
                    faces=dot.faces,
                    batched_positions=pts,
                    batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(pts), 1)),
                    batched_colors=color,
                    opacity=0.9,
                )

    # animated human stick figure (the solver's reference cloud)
    handles: list = []
    paired_segments: bool | None = None
    parent_idx = np.array([demo.index(LAFAN_PARENT[n]) if LAFAN_PARENT[n] in demo else -1 for n in demo])
    if joints is not None and args.data_format == "lafan":
        bones = [(p, c) for c, p in enumerate(parent_idx) if p >= 0]
    else:
        bones = []

    def bone_points(frame: np.ndarray, paired: bool) -> np.ndarray:
        segs = np.array([[frame[p], frame[c]] for p, c in bones])
        return segs if paired else segs.reshape(-1, 3)

    def show(frame: int) -> None:
        nonlocal paired_segments
        for handle in handles:
            handle.remove()
        handles.clear()
        f = joints[frame] * scale
        if bones:
            if paired_segments is None:
                try:
                    handle = server.scene.add_line_segments(
                        "/skeleton",
                        points=bone_points(f, paired=True),
                        colors=np.tile(BONE_COLOR, (len(bones), 2, 1)),
                        line_width=0.01,
                    )
                    paired_segments = True
                    handles.append(handle)
                except ValueError:
                    paired_segments = False
            else:
                n = 2 * len(bones)
                colors = np.tile(BONE_COLOR, ((len(bones), 2, 1) if paired_segments else (n, 1)))
                handles.append(
                    server.scene.add_line_segments(
                        "/skeleton",
                        points=bone_points(f, paired=paired_segments),
                        colors=colors,
                        line_width=0.01,
                    )
                )
        sphere = trimesh.primitives.Sphere(radius=0.02)
        handles.append(
            server.scene.add_batched_meshes_simple(
                "/joints",
                vertices=sphere.vertices,
                faces=sphere.faces,
                batched_positions=f,
                batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (f.shape[0], 1)),
                batched_colors=JOINT_COLOR,
                opacity=1.0,
            )
        )

    playing = [True]

    if joints is not None:
        @slider.on_update
        def _on_slider(_: object = slider) -> None:
            show(int(slider.value))

        @pause_btn.on_click
        def _on_pause(_: object = pause_btn) -> None:
            playing[0] = not playing[0]

        show(0)
        while True:
            if playing[0] and float(speed.value) > 0:
                nxt = (int(slider.value) + 1) % len(joints)
                slider.value = nxt
                show(nxt)
                time.sleep(1.0 / (30.0 * float(speed.value)))
            else:
                time.sleep(0.05)
    else:
        print(f"viser serving on port {args.port} (no motion npy found - terrain only)")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
