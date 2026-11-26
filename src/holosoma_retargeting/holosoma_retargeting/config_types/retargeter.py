"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int] | tuple[int, int, float]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Each window is (start, end) or (start, end, z_floor).
    If z_floor is given per-window, it overrides the global z_floor for that window.
    Example: {"L_Toe": [(30, 60, 0.15)], "R_Toe": [(10, 20), (80, 95, 0.30)]}"""

    z_floor: float = 0.0
    """Default floor height used by Z pinning constraints (overridden by per-window z)."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""

    auto_from_stance: bool = False
    """Auto-populate the lock windows from the detected human stance sequence.

    Equivalent to calling ``retargeter.auto_foot_lock_from_stance`` with the
    ``foot_sticking_sequences`` computed for the motion: every stance interval
    becomes a Z-pinning window for that side's sole/toe links. This keeps the
    foot sole on the ground when the leg chain cannot lower the ankle landmark
    to its target (robots whose limb proportions exceed the uniformly scaled
    human's), which otherwise manifests as standing on tiptoe with the heel
    unobserved by the objective.
    """

    auto_min_frames: int = 3
    """Minimum stance-interval length (frames) for an auto foot-lock window."""

    soft: bool = False
    """Apply the Z pin as a strong residual instead of a hard constraint.

    A hard one-step pin conflicts with the trust region when the foot is
    currently far above the anchor and makes the QP infeasible; the soft
    anchor pulls the foot down over successive SQP iterations (this is how
    TERRA's calcaneus anchor residual works). auto_from_stance enables this.
    """

    soft_weight: float = 2000.0
    """Weight of each soft foot-lock anchor residual."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
