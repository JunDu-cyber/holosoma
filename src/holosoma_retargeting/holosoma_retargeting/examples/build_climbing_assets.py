"""Build a holosoma climbing task directory from a raw motion file.

Runs the vendored TERRA terrain reconstruction (holosoma_retargeting.src.
terrain_reconstruction) on the motion, then emits everything the climbing
task expects, mirroring the demo_data/climb/<seq> layout:

    <task>_joint_positions.npy   raw motion (copied; the loader globs *.npy)
    multi_boxes.obj              merged terrain mesh, human scale
    box_models/boxN.obj          per-box meshes (vertices in world coords)
    box_assets.xml               MuJoCo mesh assets, scaled to the robot
    box_body.xml                 static box bodies (contype/conaffinity 1)
    multi_boxes.urdf             viser playback URDF (same robot scale)
    terrain.json                 TerrainSpec + fit report sidecar
    n2_18dof_w_multi_boxes.xml   robot scene XML (+ ground + box includes)

The scene follows the upstream demo convention (demo_data/climb/
mocap_climb_seq_*, G1 = 0.7416) in MECHANISM but not in value: every
terrain mesh carries a ``scale`` attribute so the collision geometry can
live in any world, and ``terrain.json`` records which one (``scene_scale``).
Default 1.0 - the terrain keeps its real dimensions; ``--scene_scale robot``
reproduces the upstream value (terrain shrunk to robot size). The
retargeter applies ``scene_scale`` to the collision geometry, the foot-lock
anchors and the direct placement anchors (torso + toes): the interaction
mesh tracks only the robot-scale human SHAPE, and the anchors place it on
the terrain - translated, not scaled (root-relative retargeting).

The emitted scene references box_assets.xml/box_body.xml via <include>, so
the repo's augmentation machinery (create_scaled_multi_boxes_xml, which
regex-rewrites mesh scale attributes) keeps working. Caveat: that mesh-scale
augmentation is exact for unpitched boxes; for fitted ramps it distorts the
grade slightly (TerrainSpec.scale_height would be the exact route).

Usage (from the holosoma_retargeting package root):

    python examples/build_climbing_assets.py --motion_npy <raw.npy> \
        --data_format lafan --task_name my_stairs --data_path demo_data/climb_lafan

Then retarget:

    python examples/robot_retarget.py --data_path demo_data/climb_lafan \
        --task-type climbing --task-name my_stairs --data_format lafan \
        --robot n2 --retargeter.foot-lock.auto-from-stance
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from holosoma_retargeting.config_types.data_type import DATA_FORMAT_CONSTANTS, DEMO_JOINTS_REGISTRY
from holosoma_retargeting.config_types.robot import _ROBOT_DEFAULTS  # noqa: PLC2701 - single source of robot heights
from holosoma_retargeting.src.terrain_reconstruction import BoxSpec, TerrainSpec, reconstruct_terrain


def _y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    """Proper rotation, matches utils.transform_y_up_to_z_up (x stays, z->-y, y->z)."""
    out = points.copy()
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def _box_mesh(box: BoxSpec) -> trimesh.Trimesh:
    """World-coordinates trimesh box for one BoxSpec (half-extents + yaw/pitch)."""
    mesh = trimesh.creation.box(extents=np.asarray(box.size) * 2.0)
    cy, sy = np.cos(box.yaw), np.sin(box.yaw)
    cp, sp = np.cos(box.pitch), np.sin(box.pitch)
    R = np.array([
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ])  # Rz(yaw) @ Ry(pitch), the BoxSpec.rotation convention
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(box.pos)
    mesh.apply_transform(T)
    return mesh


def _quat_str(box: BoxSpec) -> str:
    w, x, y, z = box.quat  # MuJoCo wxyz
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion_npy", required=True, help="raw motion joints (T, J, 3); lafan expects y-up")
    parser.add_argument("--task_name", default=None, help="task name (default: motion file stem)")
    parser.add_argument("--data_path", default="demo_data/climb_lafan", help="parent dir of the task directory")
    parser.add_argument("--data_format", default="lafan", choices=["lafan", "mocap"])
    parser.add_argument("--robot", default="n2", help="robot name (scene template <robot>_18dof.xml)")
    parser.add_argument(
        "--scene_scale",
        default="1.0",
        help="terrain scale in the robot world: '1.0' (default, real dimensions) or 'robot' (shrink with robot_height/1.78)",
    )
    parser.add_argument("--fps", type=float, default=None, help="source fps (default 30 lafan / 120 mocap)")
    parser.add_argument("--stair_flight", default="auto", choices=["auto", "off"])
    parser.add_argument("--ramp", default="auto", choices=["auto", "off"])
    parser.add_argument("--seat", default="auto", choices=["auto", "off"])
    parser.add_argument("--overwrite", action="store_true", help="re-emit assets into an existing task dir")
    args = parser.parse_args()

    fmt = args.data_format
    fps = args.fps if args.fps is not None else (120.0 if fmt == "mocap" else 30.0)
    downsample = 4 if fmt == "mocap" else 1  # mirrors load_motion_data's climbing branch
    task_name = args.task_name or Path(args.motion_npy).stem
    task_dir = Path(args.data_path) / task_name
    if task_dir.exists() and not args.overwrite:
        raise SystemExit(f"{task_dir} already exists (use --overwrite to re-emit)")

    # Retarget-time artifacts (*_scaled_*.xml/urdf) are derived from the raw
    # assets by create_scaled_multi_boxes_xml / create_new_scene_xml_file,
    # which short-circuit when the output already exists. If a previous
    # generation leaves them behind, the next retarget would silently keep
    # solving against the OLD collision geometry. They must not outlive the
    # raw assets they were derived from.
    stale = sorted(task_dir.glob("*_scaled_*")) if task_dir.exists() else []
    for p in stale:
        p.unlink()
    if stale:
        print(f"removed {len(stale)} stale retarget-time artifact(s): {[p.name for p in stale]}")

    raw = np.load(args.motion_npy)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise SystemExit(f"motion npy must be (T, J, 3), got {raw.shape}")
    demo = list(DEMO_JOINTS_REGISTRY[fmt])
    if raw.shape[1] != len(demo):
        raise SystemExit(f"{fmt} expects {len(demo)} joints, npy has {raw.shape[1]}")

    # scene_scale: how the terrain enters the robot world. Default 1.0 - the
    # terrain keeps its REAL dimensions (a 0.17 m stair rise stays 0.17 m for a
    # 1.22 m robot; the reference human is still scaled to robot size by the
    # retargeter, and the foot-lock anchors adapt via height_at). The older
    # convention shrank the terrain with the human (robot_height/1.78); pass
    # --scene_scale robot to reproduce it.
    if args.scene_scale == "robot":
        human_height = DATA_FORMAT_CONSTANTS.get(fmt, {}).get("default_human_height") or 1.78
        scene_scale = _ROBOT_DEFAULTS[args.robot]["robot_height"] / human_height
    else:
        scene_scale = float(args.scene_scale)
    scale_str = f"{scene_scale} {scene_scale} {scene_scale}"
    print(f"scene mesh scale: {scene_scale:.6f} (terrain dimensions x scene_scale)")

    joints = raw[::downsample].astype(float)
    if fmt == "lafan":
        joints = _y_up_to_z_up(joints)

    print(f"motion: {len(raw)} -> {len(joints)} frames at {fps / downsample:.1f} Hz")
    spec, report = reconstruct_terrain(
        joints, demo, fps / downsample, fmt,
        stair_flight=args.stair_flight, ramp=args.ramp, seat=args.seat,
    )
    warnings = report.get("warnings", [])
    for w in warnings:
        print(f"  [fit] {w}")

    if spec.is_flat:
        print("Reconstruction found NO raised terrain (flat motion). "
              "A climbing task needs stairs/ramps/boxes; nothing to emit.")
        return 1

    print(f"terrain: {len(spec.boxes)} boxes")
    for b in spec.boxes:
        pitch = f" pitch {np.degrees(b.pitch):+.1f} deg" if b.is_sloped else ""
        top = f" top {b.top:.3f}" if not b.is_sloped else ""
        print(f"  {b.name:24s} pos [{b.pos[0]:+.2f} {b.pos[1]:+.2f} {b.pos[2]:+.2f}]"
              f" size {np.round(b.size, 3).tolist()}{pitch}{top}")

    # ---------------- emit the task directory --------------------------------
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "box_models").mkdir(exist_ok=True)

    npy_dest = task_dir / f"{task_name}_joint_positions.npy"
    if not npy_dest.exists():
        shutil.copy(args.motion_npy, npy_dest)

    # merged mesh + per-box meshes (vertices carry the world placement)
    boxes = [_box_mesh(b) for b in spec.boxes]
    merged = trimesh.util.concatenate(boxes)
    merged.export(task_dir / "multi_boxes.obj")
    body_lines = ["<mujocoinclude>"]
    for i, (b, mesh) in enumerate(zip(spec.boxes, boxes), start=1):
        mesh.export(task_dir / "box_models" / f"box{i}.obj")
        body_lines.append(
            f'    <body name="multi_boxes_box{i}_link" pos="0 0 0" quat="{_quat_str(b)}">\n'
            f'        <geom name="multi_boxes_link_{i}" type="mesh" mesh="box{i}" '
            f'pos="0 0 0" quat="1 0 0 0" contype="1" conaffinity="1"/>\n'
            f"    </body>"
        )
    body_lines += ["</mujocoinclude>"]
    (task_dir / "box_body.xml").write_text("\n".join(body_lines) + "\n")

    # viser playback URDF (visual only; meshes already carry placement)
    urdf = (
        '<?xml version="1.0"?>\n'
        f'<robot name="multi_boxes">\n'
        f'  <link name="multi_boxes_link">\n'
        f'    <visual><geometry><mesh filename="./multi_boxes.obj" scale="{scale_str}"/></geometry></visual>\n'
        f'  </link>\n'
        f"</robot>\n"
    )
    (task_dir / "multi_boxes.urdf").write_text(urdf)

    # terrain sidecar consumed by setup_object_data (exact TERRA point sampling)
    with open(task_dir / "terrain.json", "w") as f:
        json.dump(
            {
                "terrain": spec.to_dict(),
                "scene_scale": scene_scale,
                "report": {k: v for k, v in report.items() if k not in ("levels",)},
                "provenance": "holosoma_retargeting.src.terrain_reconstruction (vendored TERRA @db9d0d6)",
                "source_motion": str(args.motion_npy),
            },
            f,
            indent=2,
        )
    print(f"wrote {task_dir / 'terrain.json'}")

    # robot scene: merge the robot XML with the box includes. Follows the G1
    # demo convention: meshdir points from the task dir back to the robot's
    # meshes, and the box mesh files in box_assets.xml are relative to that
    # meshdir (so the augmentation regex on file="box_assets.xml" keeps working).
    import os
    import xml.etree.ElementTree as ET

    pkg_root = Path(__file__).resolve().parents[1]
    robot_xml = pkg_root / "models" / args.robot / f"{args.robot}_18dof.xml"
    robot_meshes = pkg_root / "models" / args.robot / "meshes"
    scene_name = f"{args.robot}_18dof_w_multi_boxes.xml"
    scene_path = task_dir / scene_name

    # box_assets.xml: mesh files relative to the robot's meshdir. Raw <mesh>
    # children inside a <mujocoinclude> wrapper (no <asset> nesting - it is
    # included inside the scene's <asset> section, mirroring the G1 demo).
    mesh_rel = os.path.relpath(robot_meshes, task_dir)
    box_file = lambda i: os.path.relpath(task_dir / "box_models" / f"box{i}.obj", robot_meshes)  # noqa: E731
    asset_lines = ["<mujocoinclude>"]
    for i in range(1, len(spec.boxes) + 1):
        asset_lines.append(f'    <mesh name="box{i}" file="{box_file(i)}" scale="{scale_str}"/>')
    asset_lines += ["</mujocoinclude>"]
    (task_dir / "box_assets.xml").write_text("\n".join(asset_lines) + "\n")

    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", mesh_rel)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "include").set("file", "box_assets.xml")
    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "include").set("file", "box_body.xml")
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    print(f"wrote {scene_path}")

    print("\nRetarget with:")
    print(f"  python examples/robot_retarget.py --data_path {args.data_path} " + chr(92))
    print(f"      --task-type climbing --task-name {task_name} --data_format {fmt} " + chr(92))
    print(f"      --robot {args.robot} --retargeter.foot-lock.auto-from-stance")
    print("Playback with:")
    print(f"  python viser_player.py --robot_urdf models/{args.robot}/{args.robot}_18dof.urdf " + chr(92))
    print(f"      --object_urdf {scene_path.parent / 'multi_boxes.urdf'} " + chr(92))
    print(f"      --qpos_npz demo_results/{args.robot}/climbing/<fmt>_climb/{task_name}_original.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
