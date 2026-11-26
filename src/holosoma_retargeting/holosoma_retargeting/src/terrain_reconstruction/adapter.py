"""Bridge between holosoma's motion loaders and the vendored TERRA reconstruction.

The reconstruction core (:func:`fit_terrain_from_motion`) consumes a plain
``(T, J, 3)`` landmark array plus joint names, and expects the four foot-contact
probes to be named the SMPL-H way (``L_Toe``/``R_Toe``/``L_Ankle``/``R_Ankle``).
holosoma's loaders already emit those names for the ``smplh`` format; the other
formats use different names for the same landmarks, so this module renames the
probe columns before fitting. The renaming follows the same correspondences the
``JOINTS_MAPPINGS`` entries use when retargeting to a robot (e.g. LAFAN's
``LeftFoot`` is the ankle-side landmark, ``LeftToeBase`` the toe).

Expected input is exactly what the loaders produce *before* robot scaling:
world-frame, z-up, human-scale joint positions (for LAFAN, after the loader's
``transform_y_up_to_z_up``). The fitted boxes come out in that same world frame
at human scale, which is the convention the climbing task's ``multi_boxes.obj``
uses (``setup_object_data`` applies ``smpl_scale`` itself).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ._spec import BoxSpec, TerrainSpec
from .family import SMPLH_NEUTRAL_FOOT_PITCH_DEG
from .fitting import fit_terrain_from_motion
from .stance import DEFAULT_CONTACT_JOINTS

#: Renames for the four contact probes, per holosoma data format. ``smplh`` needs
#: none: its ``DEMO_JOINTS`` are already the names the core expects.
_FORMAT_CONTACT_ALIASES: dict[str, dict[str, str]] = {
    "smplh": {},
    "smplx": {"L_Foot": "L_Toe", "R_Foot": "R_Toe"},
    "lafan": {
        "LeftToeBase": "L_Toe",
        "RightToeBase": "R_Toe",
        "LeftFoot": "L_Ankle",
        "RightFoot": "R_Ankle",
    },
    "mocap": {
        "LeftToeBase": "L_Toe",
        "RightToeBase": "R_Toe",
        "LeftFoot": "L_Ankle",
        "RightFoot": "R_Ankle",
    },
}

_SUPPORTED_FORMATS = tuple(_FORMAT_CONTACT_ALIASES)


def _rename_probes(joint_names: Sequence[str], data_format: str) -> tuple[str, ...]:
    """Return joint names with the format's foot probes renamed to SMPL-H names."""

    aliases = _FORMAT_CONTACT_ALIASES[data_format]
    renamed = tuple(aliases.get(name, name) for name in joint_names)
    if len(set(renamed)) != len(renamed):
        raise ValueError(
            f"renaming {data_format} joints for terrain reconstruction produced "
            f"duplicate names; the input skeleton likely already contains "
            f"{sorted(set(aliases.values()))}"
        )
    return renamed


def reconstruct_terrain(
    joints: np.ndarray,
    joint_names: Sequence[str],
    fps: float,
    data_format: str,
    neutral_foot_pitch: Mapping[str, float] | None = None,
    seat: str = "auto",
    **fit_options: object,
) -> tuple[TerrainSpec, dict]:
    """Fit a :class:`TerrainSpec` from one holosoma-loaded human motion.

    Args:
        joints: ``(T, J, 3)`` world-frame, z-up, *human-scale* joint positions as
            produced by the loaders in ``examples/robot_retarget.py`` (before the
            ``smpl_scale`` step; for LAFAN, after ``transform_y_up_to_z_up``).
        joint_names: The ``DEMO_JOINTS`` list matching the array's joint axis.
        fps: Frame rate of ``joints`` in frames per second.
        data_format: One of ``smplh``, ``smplx``, ``lafan``, ``mocap``; selects the
            probe renaming table.
        neutral_foot_pitch: Left/right ankle-to-toe pitch of a foot resting on flat
            ground, keyed ``{"L": deg, "R": deg}``. ``None`` uses the per-format
            default: the fitted-SMPL-H rest pitch for ``smplh`` (TERRA's own
            convention), and no calibration for the other formats, in which case
            family selection falls back to the footfall-profile residual.
        seat: ``"auto"`` or ``"off"``. Seat inference additionally requires a
            ``Pelvis`` joint in the skeleton; ``"auto"`` is demoted to ``"off"``
            when it is absent (e.g. LAFAN/mocap joint lists).
        **fit_options: Forwarded verbatim to :func:`fit_terrain_from_motion`
            (``stance_speed``, ``level_tol``, ``stair_flight``, ``ramp``, ...).

    Returns:
        ``(terrain, report)``. ``terrain.is_flat`` is a legitimate result for a
        motion on flat ground. ``report`` carries stance/level/fit diagnostics.

    Raises:
        ValueError: If ``data_format`` is unknown or the inputs are malformed.
    """
    if data_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unknown data_format {data_format!r}; supported: {', '.join(_SUPPORTED_FORMATS)}"
        )
    motion = np.asarray(joints, dtype=float)
    names = _rename_probes(joint_names, data_format)

    if neutral_foot_pitch is None and data_format == "smplh":
        neutral_foot_pitch = dict(SMPLH_NEUTRAL_FOOT_PITCH_DEG)
    if seat == "auto" and "Pelvis" not in names:
        seat = "off"

    missing = [name for name in DEFAULT_CONTACT_JOINTS if name not in names]
    if missing:
        raise ValueError(
            f"after {data_format} renaming the skeleton lacks the contact probes {missing}; "
            "the motion cannot support terrain reconstruction"
        )

    return fit_terrain_from_motion(
        motion,
        names,
        fps,
        neutral_foot_pitch=neutral_foot_pitch,
        seat=seat,
        **fit_options,
    )


__all__ = ["reconstruct_terrain", "BoxSpec", "TerrainSpec"]
