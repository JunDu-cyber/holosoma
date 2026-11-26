"""Terrain reconstruction vendored from TERRA (amathislab/terra, Apache-2.0).

Reconstructs motion-supported box terrain (stair flights, ramps, platforms, seats)
from plain kinematic landmark trajectories, following

    Simos, M., Li, C., Ziliotto, B., & Mathis, A. (2026).
    "TERRA: Terrain-Aware Reconstruction, Retargeting and Control for
    Musculoskeletal Locomotion." arXiv:2609.38653.

Provenance
----------
* ``_spec.py``            -- amathislab/musclemimic @ f1c2dbf,
                             ``loco_mujoco/core/terrain/boxes.py`` (loco_mujoco,
                             Apache-2.0): ``BoxSpec`` / ``TerrainSpec`` geometry.
* every other module      -- amathislab/terra @ db9d0d6,
                             ``src/terra/terrain/<name>.py`` (Apache-2.0).
* TERRA's ``NOTICE`` credits adapted code in its own tree (including holosoma's
  retargeter Jacobians); that attribution chain carries over with this copy.

These are pinned snapshots: upstream changes do not propagate. The only edits are
import rewrites (``terra._musclemimic`` -> ``._spec``, ``terra.terrain`` ->
relative, ``terra._files.atomic_write`` -> ``._compat``) and the removal of the
MuJoCo/JAX env-side ``BoxTerrain`` plugin, which the retargeting pipeline does not
use. The result is numpy-only: no torch, no MuJoCo, no SMPL-H model files.

Entry point for the holosoma pipeline: :func:`adapter.reconstruct_terrain`, which
accepts the joint arrays and joint-name lists produced by the existing
``robot_retarget.py`` motion loaders.
"""

from ._spec import BoxSpec, TerrainSpec
from .adapter import reconstruct_terrain
from .family import SMPLH_NEUTRAL_FOOT_PITCH_DEG
from .fitting import fit_terrain_from_motion
from .metadata import TerrainMetadata
from .stance import DEFAULT_CONTACT_JOINTS

__all__ = [
    "BoxSpec",
    "TerrainSpec",
    "TerrainMetadata",
    "DEFAULT_CONTACT_JOINTS",
    "SMPLH_NEUTRAL_FOOT_PITCH_DEG",
    "fit_terrain_from_motion",
    "reconstruct_terrain",
]
