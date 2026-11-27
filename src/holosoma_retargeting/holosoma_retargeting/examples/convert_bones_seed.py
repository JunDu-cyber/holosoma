#!/usr/bin/env python3
"""Convert a bones-seed kimodo SOMA77 npz to a lafan-format motion npy.

The bones-seed dataset (curated SOMA mocap, e.g.
/data/dj/datasets/bones-seed/kimodo_npz) stores each clip as a npz produced
by ``kimodo_convert`` with:

- ``posed_joints``  (T, 77, 3)  world positions, meters, y-up, 30 fps
- ``root_positions``/``foot_contacts`` etc. (unused here)

The retargeting pipeline expects a lafan npy: (T, 22, 3) world joints in
``LAFAN_DEMO_JOINTS`` order, y-up. The SOMA77 joint order equals the BVH
depth-first parse order (verified: index 0 == Hips == ``root_positions``,
anatomy of Head/HeadEnd/eyes/toes is self-consistent). soma_uniform is a
single normalized rig, so every clip shares one body proportion.

Usage:
    python examples/convert_bones_seed.py <clip.npz> [<clip2.npz> ...] \
        [--out_dir demo_data/climb_lafan]

Writes <clip-stem>.npy next to the input (or into --out_dir).
"""

import argparse
from pathlib import Path

import numpy as np

# SOMA77 joint order == BVH parse order of the soma_uniform skeleton.
SOMA77_JOINTS = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "HeadEnd", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]

# lafan joint -> single soma joint, or a list to average (virtual joints).
LAFAN_TO_SOMA = {
    "Hips": ["Hips"],
    "RightUpLeg": ["RightLeg"],
    "RightLeg": ["RightShin"],
    "RightFoot": ["RightFoot"],
    "RightToeBase": ["RightToeBase"],
    "LeftUpLeg": ["LeftLeg"],
    "LeftLeg": ["LeftShin"],
    "LeftFoot": ["LeftFoot"],
    "LeftToeBase": ["LeftToeBase"],
    # soma's torso chain (Spine1, Spine2, Chest) maps onto lafan's
    # (Spine, Spine1, Spine2) ascending; lafan has a single Neck so use the
    # midpoint of Neck1/Neck2.
    "Spine": ["Spine1"],
    "Spine1": ["Spine2"],
    "Spine2": ["Chest"],
    "Neck": ["Neck1", "Neck2"],
    "Head": ["Head"],
    "RightShoulder": ["RightShoulder"],
    "RightArm": ["RightArm"],
    "RightForeArm": ["RightForeArm"],
    "RightHand": ["RightHand"],
    "LeftShoulder": ["LeftShoulder"],
    "LeftArm": ["LeftArm"],
    "LeftForeArm": ["LeftForeArm"],
    "LeftHand": ["LeftHand"],
}


def convert(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    posed = np.asarray(data["posed_joints"], dtype=np.float64)
    if posed.ndim != 3 or posed.shape[1] != len(SOMA77_JOINTS):
        raise ValueError(
            f"{npz_path}: posed_joints shape {posed.shape}, expected "
            f"(T, {len(SOMA77_JOINTS)}, 3)"
        )
    soma_idx = {n: i for i, n in enumerate(SOMA77_JOINTS)}

    out = np.empty((posed.shape[0], len(LAFAN_TO_SOMA), 3), dtype=np.float64)
    for row, (lafan_name, soma_names) in enumerate(LAFAN_TO_SOMA.items()):
        out[:, row, :] = posed[:, [soma_idx[n] for n in soma_names], :].mean(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="kimodo SOMA77 npz file(s)")
    parser.add_argument("--out_dir", type=Path, default=None, help="output directory (default: beside each input)")
    args = parser.parse_args()

    for src in args.inputs:
        joints = convert(src)
        out_dir = args.out_dir or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{src.stem}.npy"
        np.save(dst, joints)

        # sanity report: y-up in meters, feet near the floor at some point
        hips = joints[:, 0, 1]
        toe_min = min(joints[:, 4, 1].min(), joints[:, 8, 1].min())
        print(
            f"{src.name} -> {dst}\n"
            f"    frames {joints.shape[0]} (30 fps = {joints.shape[0] / 30:.1f} s), "
            f"hips y {hips.min():.2f}..{hips.max():.2f} m, "
            f"lowest toe {toe_min:.3f} m, highest toe "
            f"{max(joints[:, 4, 1].max(), joints[:, 8, 1].max()):.2f} m"
        )


if __name__ == "__main__":
    main()
