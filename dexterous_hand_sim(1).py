"""
Dexterous Hand Kinematic Simulation
===================================

A single-file Python 3.12 project for a biomimetic dexterous hand skeleton
simulation using only NumPy and Matplotlib 3D.

Run:
    python dexterous_hand_sim.py

Self-test:
    python dexterous_hand_sim.py --self-test

Keyboard controls in the animation window:
    n       next pose
    p       previous pose
    space   pause / resume
    r       reset to open hand
    q/esc   quit
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Tuple

import numpy as np


# =============================================================================
# 1. Global configuration
# =============================================================================

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
FOUR_FINGER_NAMES = ["index", "middle", "ring", "little"]

FOUR_FINGER_JOINTS = ["MCP", "PIP", "DIP", "TIP"]
THUMB_JOINTS = ["CMC", "MCP", "IP", "TIP"]
ROOT_JOINTS = ["FOREARM_BASE", "WRIST"]

LOCAL_FORWARD = np.array([0.0, 1.0, 0.0], dtype=float)
ZERO3 = np.zeros(3, dtype=float)
FOREARM_BASE_LOCAL = np.array([0.0, -55.0, 0.0], dtype=float)

WRIST_LIMITS = {
    "wrist_flex_ext": (-60.0, 70.0),
    "wrist_radial_ulnar": (-20.0, 30.0),
    "forearm_prono_supination": (-80.0, 80.0),
}

FOUR_FINGER_LIMITS = {
    "mcp_abd_add": (-15.0, 15.0),
    "mcp_flex": (0.0, 90.0),
    "pip_flex": (0.0, 110.0),
    "dip_flex": (0.0, 80.0),
}

THUMB_LIMITS = {
    "cmc_abd_add": (-20.0, 35.0),
    "cmc_flex": (0.0, 45.0),
    "cmc_axial_rot": (0.0, 60.0),
    "mcp_abd_add": (-15.0, 15.0),
    "mcp_flex": (0.0, 55.0),
    "ip_flex": (0.0, 80.0),
}

BONE_LENGTHS = {
    "thumb": {
        "CMC_MCP": 32.0,
        "MCP_IP": 25.0,
        "IP_TIP": 20.0,
    },
    "index": {
        "MCP_PIP": 42.0,
        "PIP_DIP": 25.0,
        "DIP_TIP": 18.0,
    },
    "middle": {
        "MCP_PIP": 47.0,
        "PIP_DIP": 28.0,
        "DIP_TIP": 20.0,
    },
    "ring": {
        "MCP_PIP": 44.0,
        "PIP_DIP": 26.0,
        "DIP_TIP": 19.0,
    },
    "little": {
        "MCP_PIP": 35.0,
        "PIP_DIP": 21.0,
        "DIP_TIP": 16.0,
    },
}

PALM_BASE_POINTS = {
    "thumb": np.array([-28.0, 18.0, 0.0], dtype=float),
    "index": np.array([-18.0, 48.0, 0.0], dtype=float),
    "middle": np.array([-6.0, 52.0, 0.0], dtype=float),
    "ring": np.array([7.0, 49.0, 0.0], dtype=float),
    "little": np.array([19.0, 43.0, 0.0], dtype=float),
}

FINGER_BASE_DIRECTIONS = {
    "thumb": np.array([-0.5, 0.8660254, 0.0], dtype=float),
    "index": np.array([0.0, 1.0, 0.0], dtype=float),
    "middle": np.array([0.0, 1.0, 0.0], dtype=float),
    "ring": np.array([0.0, 1.0, 0.0], dtype=float),
    "little": np.array([0.0, 1.0, 0.0], dtype=float),
}

BONE_CONNECTIONS = [
    ("root", "FOREARM_BASE", "root", "WRIST"),
    ("thumb", "CMC", "thumb", "MCP"),
    ("thumb", "MCP", "thumb", "IP"),
    ("thumb", "IP", "thumb", "TIP"),
    ("index", "MCP", "index", "PIP"),
    ("index", "PIP", "index", "DIP"),
    ("index", "DIP", "index", "TIP"),
    ("middle", "MCP", "middle", "PIP"),
    ("middle", "PIP", "middle", "DIP"),
    ("middle", "DIP", "middle", "TIP"),
    ("ring", "MCP", "ring", "PIP"),
    ("ring", "PIP", "ring", "DIP"),
    ("ring", "DIP", "ring", "TIP"),
    ("little", "MCP", "little", "PIP"),
    ("little", "PIP", "little", "DIP"),
    ("little", "DIP", "little", "TIP"),
]

PALM_CONNECTIONS = [
    ("root", "WRIST", "thumb", "CMC"),
    ("root", "WRIST", "index", "MCP"),
    ("root", "WRIST", "middle", "MCP"),
    ("root", "WRIST", "ring", "MCP"),
    ("root", "WRIST", "little", "MCP"),
    ("thumb", "CMC", "index", "MCP"),
    ("index", "MCP", "middle", "MCP"),
    ("middle", "MCP", "ring", "MCP"),
    ("ring", "MCP", "little", "MCP"),
]

JOINT_COLORS = {
    "FOREARM_BASE": "tab:gray",
    "WRIST": "black",
    "CMC": "tab:orange",
    "MCP": "tab:blue",
    "PIP": "tab:green",
    "DIP": "tab:red",
    "IP": "tab:purple",
    "TIP": "tab:pink",
}

BONE_COLOR = "dimgray"
PALM_BONE_COLOR = "lightgray"

FPS = 30
TRANSITION_FRAMES = 24
HOLD_FRAMES = 40
PHASE_FRAMES = 90
PHASE_CYCLES_PER_ACTION = 2


AngleDict = Dict[str, Dict[str, float]]
JointDict = Dict[str, Dict[str, np.ndarray]]
PoseFn = Callable[[], AngleDict]
PhasePoseFn = Callable[[float], AngleDict]


def get_pyplot():
    """Lazy-import matplotlib.pyplot so self-tests can run without loading the GUI stack."""
    import matplotlib.pyplot as plt
    return plt


# =============================================================================
# 2. Rotation matrix utilities
# =============================================================================


def deg2rad(angle_deg: float) -> float:
    """Convert degrees to radians."""
    return float(np.deg2rad(angle_deg))


def rot_x(angle_rad: float) -> np.ndarray:
    """Rotation around the X axis.

    With the coordinate convention used here, a positive X rotation maps local
    +Y toward local +Z, so positive finger flexion bends toward the palm.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rot_y(angle_rad: float) -> np.ndarray:
    """Rotation around the Y axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    """Rotation around the Z axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized vector; raise if the vector is nearly zero."""
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def rot_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotation around an arbitrary axis using Rodrigues' formula."""
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def align_vector_to_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps direction src to direction dst."""
    src_n = normalize(src)
    dst_n = normalize(dst)
    cross = np.cross(src_n, dst_n)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(src_n, dst_n), -1.0, 1.0))

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=float)
        # 180 degree turn. Choose a stable axis perpendicular to src.
        candidate = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(candidate, src_n))) > 0.9:
            candidate = np.array([0.0, 0.0, 1.0], dtype=float)
        axis = normalize(np.cross(src_n, candidate))
        return rot_axis(axis, math.pi)

    angle = math.atan2(cross_norm, dot)
    return rot_axis(cross / cross_norm, angle)


# =============================================================================
# 3. Default angles, limit checks, and clamping
# =============================================================================


def neutral_angles() -> AngleDict:
    """Return the default open-hand angle dictionary.

    All angles are stored in degrees. The thumb's natural 30-degree spread is
    represented by FINGER_BASE_DIRECTIONS["thumb"], not by a non-zero CMC angle.
    """
    return {
        "root": {
            "wrist_flex_ext": 0.0,
            "wrist_radial_ulnar": 0.0,
            "forearm_prono_supination": 0.0,
        },
        "thumb": {
            "cmc_abd_add": 0.0,
            "cmc_flex": 0.0,
            "cmc_axial_rot": 0.0,
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "ip_flex": 0.0,
        },
        "index": {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        },
        "middle": {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        },
        "ring": {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        },
        "little": {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        },
    }


def pose_open() -> AngleDict:
    """Open palm pose."""
    angles = neutral_angles()
    # These signs match the visual convention used in the design document:
    # index spreads slightly toward the thumb side, little finger toward ulnar side.
    angles["index"]["mcp_abd_add"] = -4.0
    angles["middle"]["mcp_abd_add"] = 0.0
    angles["ring"]["mcp_abd_add"] = 2.0
    angles["little"]["mcp_abd_add"] = 5.0
    return angles


def limits_for_group(group_name: str) -> Mapping[str, Tuple[float, float]]:
    if group_name == "root":
        return WRIST_LIMITS
    if group_name == "thumb":
        return THUMB_LIMITS
    if group_name in FOUR_FINGER_NAMES:
        return FOUR_FINGER_LIMITS
    raise KeyError(f"Unknown angle group: {group_name}")


def clamp_value(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), lo), hi))


def clamp_angles(angle_dict: Mapping[str, Mapping[str, float]]) -> AngleDict:
    """Return a deep-clamped copy of an angle dictionary."""
    result = copy.deepcopy(angle_dict)
    for group_name, channels in result.items():
        if group_name not in FINGER_NAMES and group_name != "root":
            continue
        limits = limits_for_group(group_name)
        for channel_name, channel_value in channels.items():
            if channel_name in limits:
                lo, hi = limits[channel_name]
                channels[channel_name] = clamp_value(channel_value, lo, hi)
    return result  # type: ignore[return-value]


def validate_angle_limits(angle_dict: Mapping[str, Mapping[str, float]], *, verbose: bool = False) -> bool:
    """Check whether every channel is inside its configured range."""
    ok = True
    for group_name, channels in angle_dict.items():
        if group_name not in FINGER_NAMES and group_name != "root":
            continue
        limits = limits_for_group(group_name)
        for channel_name, value in channels.items():
            if channel_name not in limits:
                continue
            lo, hi = limits[channel_name]
            inside = lo <= float(value) <= hi
            if not inside:
                ok = False
                if verbose:
                    print(
                        f"[angle-limit] {group_name}.{channel_name}="
                        f"{value:.3f} outside [{lo:.3f}, {hi:.3f}]"
                    )
    return ok


# =============================================================================
# 4. Forward kinematics
# =============================================================================


def get_root_transform(root_angles: Mapping[str, float]) -> np.ndarray:
    """Return the 3x3 root rotation matrix.

    Rotation order:
        forearm_prono_supination -> wrist_radial_ulnar -> wrist_flex_ext
    """
    flex = deg2rad(root_angles.get("wrist_flex_ext", 0.0))
    radial = deg2rad(root_angles.get("wrist_radial_ulnar", 0.0))
    prono = deg2rad(root_angles.get("forearm_prono_supination", 0.0))
    return rot_y(prono) @ rot_z(radial) @ rot_x(flex)


def compute_four_finger_joints(
    finger_name: str,
    finger_angles: Mapping[str, float],
    root_R: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute MCP, PIP, DIP, TIP points for a non-thumb finger."""
    if finger_name not in FOUR_FINGER_NAMES:
        raise ValueError(f"Not a four-finger name: {finger_name}")

    lengths = BONE_LENGTHS[finger_name]
    mcp = root_R @ PALM_BASE_POINTS[finger_name]

    base_align_R = align_vector_to_vector(LOCAL_FORWARD, FINGER_BASE_DIRECTIONS[finger_name])

    abd = deg2rad(finger_angles.get("mcp_abd_add", 0.0))
    mcp_flex = deg2rad(finger_angles.get("mcp_flex", 0.0))
    pip_flex = deg2rad(finger_angles.get("pip_flex", 0.0))
    dip_flex = deg2rad(finger_angles.get("dip_flex", 0.0))

    # Use -abd so the open-pose signs proposed in the design document produce
    # a visually natural spread under the standard right-handed rot_z matrix.
    base_R = root_R @ base_align_R @ rot_z(-abd)
    proximal_R = base_R @ rot_x(mcp_flex)
    middle_R = proximal_R @ rot_x(pip_flex)
    distal_R = middle_R @ rot_x(dip_flex)

    pip = mcp + proximal_R @ LOCAL_FORWARD * lengths["MCP_PIP"]
    dip = pip + middle_R @ LOCAL_FORWARD * lengths["PIP_DIP"]
    tip = dip + distal_R @ LOCAL_FORWARD * lengths["DIP_TIP"]

    return {
        "MCP": mcp,
        "PIP": pip,
        "DIP": dip,
        "TIP": tip,
    }


def compute_thumb_joints(
    thumb_angles: Mapping[str, float],
    root_R: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute CMC, MCP, IP, TIP points for the thumb."""
    lengths = BONE_LENGTHS["thumb"]
    cmc = root_R @ PALM_BASE_POINTS["thumb"]

    thumb_base_align_R = align_vector_to_vector(LOCAL_FORWARD, FINGER_BASE_DIRECTIONS["thumb"])

    cmc_abd = deg2rad(thumb_angles.get("cmc_abd_add", 0.0))
    cmc_flex = deg2rad(thumb_angles.get("cmc_flex", 0.0))
    cmc_axial = deg2rad(thumb_angles.get("cmc_axial_rot", 0.0))
    mcp_abd = deg2rad(thumb_angles.get("mcp_abd_add", 0.0))
    mcp_flex = deg2rad(thumb_angles.get("mcp_flex", 0.0))
    ip_flex = deg2rad(thumb_angles.get("ip_flex", 0.0))

    cmc_R = root_R @ thumb_base_align_R
    cmc_R = cmc_R @ rot_z(cmc_abd)
    cmc_R = cmc_R @ rot_x(cmc_flex)

    # Rotate the current local thumb frame around its present metacarpal axis.
    # Pre-multiplication applies the Rodrigues rotation in world coordinates.
    current_thumb_axis = cmc_R @ LOCAL_FORWARD
    cmc_R = rot_axis(current_thumb_axis, cmc_axial) @ cmc_R

    mcp_R = cmc_R @ rot_z(mcp_abd) @ rot_x(mcp_flex)
    ip_R = mcp_R @ rot_x(ip_flex)

    mcp = cmc + cmc_R @ LOCAL_FORWARD * lengths["CMC_MCP"]
    ip = mcp + mcp_R @ LOCAL_FORWARD * lengths["MCP_IP"]
    tip = ip + ip_R @ LOCAL_FORWARD * lengths["IP_TIP"]

    return {
        "CMC": cmc,
        "MCP": mcp,
        "IP": ip,
        "TIP": tip,
    }


def get_hand_joints(angle_dict: Mapping[str, Mapping[str, float]]) -> JointDict:
    """Compute all named hand joints from an angle dictionary."""
    angles = clamp_angles(angle_dict)
    root_R = get_root_transform(angles["root"])

    joints: JointDict = {
        "root": {
            "FOREARM_BASE": FOREARM_BASE_LOCAL.copy(),
            "WRIST": ZERO3.copy(),
        }
    }
    joints["thumb"] = compute_thumb_joints(angles["thumb"], root_R)
    for finger_name in FOUR_FINGER_NAMES:
        joints[finger_name] = compute_four_finger_joints(finger_name, angles[finger_name], root_R)
    return joints


def flatten_joints(hand_joints: Mapping[str, Mapping[str, np.ndarray]]) -> Tuple[list[str], np.ndarray]:
    """Flatten nested joint dict into names and an (N, 3) array."""
    names: list[str] = []
    points: list[np.ndarray] = []
    for group_name in ["root", *FINGER_NAMES]:
        for joint_name, point in hand_joints[group_name].items():
            names.append(f"{group_name}.{joint_name}")
            points.append(np.asarray(point, dtype=float))
    return names, np.vstack(points)


# =============================================================================
# 5. Pose definitions
# =============================================================================


def set_four_finger_curl(
    angles: MutableMapping[str, MutableMapping[str, float]],
    finger_name: str,
    *,
    mcp: float,
    pip: float,
    dip: float,
    abd: float | None = None,
) -> None:
    """Set curl channels for one non-thumb finger in-place."""
    if finger_name not in FOUR_FINGER_NAMES:
        raise ValueError(f"Not a four-finger name: {finger_name}")
    angles[finger_name]["mcp_flex"] = float(mcp)
    angles[finger_name]["pip_flex"] = float(pip)
    angles[finger_name]["dip_flex"] = float(dip)
    if abd is not None:
        angles[finger_name]["mcp_abd_add"] = float(abd)


def curl_four_fingers(
    angles: MutableMapping[str, MutableMapping[str, float]],
    *,
    mcp: float,
    pip: float,
    dip: float,
) -> None:
    """Apply the same curl roughly to all four non-thumb fingers."""
    for finger_name in FOUR_FINGER_NAMES:
        set_four_finger_curl(angles, finger_name, mcp=mcp, pip=pip, dip=dip)


def pose_ok() -> AngleDict:
    """OK gesture, visually approximated without inverse kinematics."""
    angles = pose_open()
    angles["thumb"].update(
        {
            "cmc_abd_add": 4.0,
            "cmc_flex": 25.0,
            "cmc_axial_rot": 35.0,
            "mcp_abd_add": 3.0,
            "mcp_flex": 20.0,
            "ip_flex": 25.0,
        }
    )
    set_four_finger_curl(angles, "index", mcp=35.0, pip=45.0, dip=25.0, abd=-2.0)
    set_four_finger_curl(angles, "middle", mcp=10.0, pip=16.0, dip=8.0, abd=0.0)
    set_four_finger_curl(angles, "ring", mcp=14.0, pip=20.0, dip=10.0, abd=2.0)
    set_four_finger_curl(angles, "little", mcp=18.0, pip=24.0, dip=12.0, abd=5.0)
    return clamp_angles(angles)


def pose_precision_grasp() -> AngleDict:
    """Precision grasp: thumb opposes index/middle, ring/little relax."""
    angles = pose_open()
    angles["thumb"].update(
        {
            "cmc_abd_add": 8.0,
            "cmc_flex": 30.0,
            "cmc_axial_rot": 45.0,
            "mcp_abd_add": 2.0,
            "mcp_flex": 25.0,
            "ip_flex": 18.0,
        }
    )
    set_four_finger_curl(angles, "index", mcp=25.0, pip=35.0, dip=18.0, abd=-1.0)
    set_four_finger_curl(angles, "middle", mcp=22.0, pip=30.0, dip=15.0, abd=0.0)
    set_four_finger_curl(angles, "ring", mcp=18.0, pip=24.0, dip=12.0, abd=2.0)
    set_four_finger_curl(angles, "little", mcp=22.0, pip=28.0, dip=14.0, abd=5.0)
    return clamp_angles(angles)


def pose_fist() -> AngleDict:
    """Closed fist pose."""
    angles = neutral_angles()
    angles["thumb"].update(
        {
            "cmc_abd_add": -8.0,
            "cmc_flex": 32.0,
            "cmc_axial_rot": 48.0,
            "mcp_abd_add": 0.0,
            "mcp_flex": 42.0,
            "ip_flex": 52.0,
        }
    )
    set_four_finger_curl(angles, "index", mcp=74.0, pip=96.0, dip=58.0, abd=0.0)
    set_four_finger_curl(angles, "middle", mcp=78.0, pip=102.0, dip=62.0, abd=0.0)
    set_four_finger_curl(angles, "ring", mcp=76.0, pip=100.0, dip=60.0, abd=0.0)
    set_four_finger_curl(angles, "little", mcp=72.0, pip=94.0, dip=56.0, abd=0.0)
    return clamp_angles(angles)


def pose_half_fist() -> AngleDict:
    """Half-closed fist pose."""
    angles = pose_open()
    angles["thumb"].update(
        {
            "cmc_abd_add": 0.0,
            "cmc_flex": 18.0,
            "cmc_axial_rot": 30.0,
            "mcp_flex": 18.0,
            "ip_flex": 16.0,
        }
    )
    set_four_finger_curl(angles, "index", mcp=36.0, pip=48.0, dip=25.0, abd=-2.0)
    set_four_finger_curl(angles, "middle", mcp=40.0, pip=54.0, dip=28.0, abd=0.0)
    set_four_finger_curl(angles, "ring", mcp=38.0, pip=52.0, dip=27.0, abd=2.0)
    set_four_finger_curl(angles, "little", mcp=34.0, pip=46.0, dip=24.0, abd=5.0)
    return clamp_angles(angles)


def pose_thumb_up() -> AngleDict:
    """Thumb-up gesture: four fingers form a fist, thumb stays extended."""
    angles = pose_fist()
    angles["root"].update(
        {
            "wrist_flex_ext": -8.0,
            "wrist_radial_ulnar": -10.0,
            "forearm_prono_supination": 35.0,
        }
    )
    angles["thumb"].update(
        {
            "cmc_abd_add": 32.0,
            "cmc_flex": 4.0,
            "cmc_axial_rot": 8.0,
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "ip_flex": 0.0,
        }
    )
    return clamp_angles(angles)


def pose_number_1() -> AngleDict:
    """Number 1: index finger up, others curled."""
    angles = pose_fist()
    angles["index"].update(
        {
            "mcp_abd_add": -4.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        }
    )
    angles["thumb"].update({"cmc_flex": 28.0, "cmc_axial_rot": 38.0, "mcp_flex": 25.0, "ip_flex": 28.0})
    return clamp_angles(angles)


def pose_number_2() -> AngleDict:
    """Number 2: index and middle fingers up, others curled."""
    angles = pose_number_1()
    angles["middle"].update(
        {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        }
    )
    return clamp_angles(angles)


def pose_number_3() -> AngleDict:
    """Number 3: index, middle, and ring fingers up; thumb/little curled."""
    angles = pose_number_2()
    angles["ring"].update(
        {
            "mcp_abd_add": 2.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        }
    )
    return clamp_angles(angles)


def pose_number_5() -> AngleDict:
    """Number 5 equals the open palm pose."""
    return pose_open()


def pose_little_up() -> AngleDict:
    """Little finger extended, other fingers curled."""
    angles = pose_fist()
    angles["little"].update(
        {
            "mcp_abd_add": 5.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        }
    )
    angles["thumb"].update({"cmc_flex": 24.0, "cmc_axial_rot": 36.0, "mcp_flex": 20.0, "ip_flex": 22.0})
    return clamp_angles(angles)


def pose_finger_wave_phase(phase: float) -> AngleDict:
    """Sequential four-finger curl/extend animation pose.

    phase should be in [0, 1], but the function also handles arbitrary values by
    wrapping them periodically.
    """
    angles = pose_open()
    base_phase = float(phase) % 1.0
    delays = {
        "index": 0.00,
        "middle": 0.18,
        "ring": 0.36,
        "little": 0.54,
    }
    for finger_name in FOUR_FINGER_NAMES:
        local_phase = (base_phase - delays[finger_name]) % 1.0
        curl = 0.5 - 0.5 * math.cos(2.0 * math.pi * local_phase)
        set_four_finger_curl(
            angles,
            finger_name,
            mcp=48.0 * curl,
            pip=70.0 * curl,
            dip=38.0 * curl,
        )
    return clamp_angles(angles)


def pose_thumb_opposition_phase(phase: float) -> AngleDict:
    """Thumb opposition animation pose."""
    angles = pose_open()
    phase = float(phase) % 1.0
    s = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)

    angles["thumb"].update(
        {
            "cmc_abd_add": -6.0 + 16.0 * s,
            "cmc_flex": 4.0 + 34.0 * s,
            "cmc_axial_rot": 4.0 + 52.0 * s,
            "mcp_abd_add": -3.0 + 6.0 * s,
            "mcp_flex": 4.0 + 26.0 * s,
            "ip_flex": 2.0 + 28.0 * s,
        }
    )

    # Slightly close index/middle as the thumb approaches opposition.
    set_four_finger_curl(angles, "index", mcp=10.0 * s, pip=18.0 * s, dip=9.0 * s, abd=-4.0)
    set_four_finger_curl(angles, "middle", mcp=8.0 * s, pip=12.0 * s, dip=6.0 * s, abd=0.0)
    return clamp_angles(angles)


def pose_wave_phase(phase: float) -> AngleDict:
    """Waving animation pose driven mainly by wrist/forearm channels."""
    angles = pose_open()
    phase = float(phase) % 1.0
    w = math.sin(2.0 * math.pi * phase)
    w2 = math.sin(2.0 * math.pi * phase + math.pi / 5.0)
    angles["root"].update(
        {
            "wrist_flex_ext": 4.0 * w2,
            "wrist_radial_ulnar": 24.0 * w,
            "forearm_prono_supination": 18.0 * w2,
        }
    )
    # A small finger spread oscillation makes the gesture less rigid.
    angles["index"]["mcp_abd_add"] = -4.0 - 1.5 * w
    angles["ring"]["mcp_abd_add"] = 2.0 + 1.0 * w
    angles["little"]["mcp_abd_add"] = 5.0 + 1.5 * w
    return clamp_angles(angles)


POSE_LIBRARY: Dict[str, PoseFn] = {
    "摊开手掌": pose_open,
    "OK手势": pose_ok,
    "精准抓握": pose_precision_grasp,
    "握拳": pose_fist,
    "竖大拇指": pose_thumb_up,
    "数字1": pose_number_1,
    "数字2": pose_number_2,
    "数字3": pose_number_3,
    "数字5": pose_number_5,
    "竖小拇指": pose_little_up,
    "半握拳": pose_half_fist,
}

PHASE_POSE_LIBRARY: Dict[str, PhasePoseFn] = {
    "手指依次屈伸": pose_finger_wave_phase,
    "拇指对掌运动": pose_thumb_opposition_phase,
    "挥手动作": pose_wave_phase,
}


# =============================================================================
# 6. Interpolation
# =============================================================================


def smoothstep(t: float) -> float:
    t = clamp_value(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def interpolate_angles(
    start_angles: Mapping[str, Mapping[str, float]],
    end_angles: Mapping[str, Mapping[str, float]],
    t: float,
    *,
    smooth: bool = True,
) -> AngleDict:
    """Interpolate every channel in two angle dictionaries."""
    u = smoothstep(t) if smooth else clamp_value(t, 0.0, 1.0)
    result = copy.deepcopy(start_angles)

    groups = set(start_angles.keys()) | set(end_angles.keys())
    for group_name in groups:
        result.setdefault(group_name, {})
        start_group = start_angles.get(group_name, {})
        end_group = end_angles.get(group_name, {})
        channels = set(start_group.keys()) | set(end_group.keys())
        for channel_name in channels:
            a = float(start_group.get(channel_name, 0.0))
            b = float(end_group.get(channel_name, 0.0))
            result[group_name][channel_name] = (1.0 - u) * a + u * b
    return clamp_angles(result)


# =============================================================================
# 7. Rendering
# =============================================================================


def configure_axes(ax, action_name: str = "", paused: bool = False) -> None:
    """Configure a 3D axis after clearing it."""
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(-70.0, 155.0)
    ax.set_zlim(-70.0, 115.0)
    ax.set_xlabel("X  thumb side → little side")
    ax.set_ylabel("Y  wrist → fingertips")
    ax.set_zlabel("Z  palm direction")
    title = f"Dexterous Hand Kinematic Simulation | {action_name}"
    if paused:
        title += " | paused"
    ax.set_title(title)
    try:
        ax.set_box_aspect((180.0, 225.0, 185.0))
    except Exception:
        pass
    ax.view_init(elev=24.0, azim=-62.0)
    ax.grid(True)


def init_3d_plot():
    """Initialize the Matplotlib 3D figure and axes."""
    plt = get_pyplot()
    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92)
    configure_axes(ax, "摊开手掌")
    fig.text(
        0.02,
        0.02,
        "Keys: n next | p previous | space pause/resume | r reset | q/esc quit",
        fontsize=9,
    )
    return fig, ax


def get_point(hand_joints: Mapping[str, Mapping[str, np.ndarray]], group_name: str, joint_name: str) -> np.ndarray:
    return np.asarray(hand_joints[group_name][joint_name], dtype=float)


def plot_connection(
    ax,
    hand_joints: Mapping[str, Mapping[str, np.ndarray]],
    connection: Tuple[str, str, str, str],
    *,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    g1, j1, g2, j2 = connection
    p1 = get_point(hand_joints, g1, j1)
    p2 = get_point(hand_joints, g2, j2)
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )


def draw_hand(ax, hand_joints: Mapping[str, Mapping[str, np.ndarray]], action_name: str = "", *, paused: bool = False) -> None:
    """Clear and redraw the hand skeleton."""
    ax.clear()
    configure_axes(ax, action_name, paused=paused)

    for connection in PALM_CONNECTIONS:
        plot_connection(
            ax,
            hand_joints,
            connection,
            color=PALM_BONE_COLOR,
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
        )

    for connection in BONE_CONNECTIONS:
        plot_connection(
            ax,
            hand_joints,
            connection,
            color=BONE_COLOR,
            linewidth=3.0,
            alpha=1.0,
            linestyle="-",
        )

    for group_name in ["root", *FINGER_NAMES]:
        for joint_name, point in hand_joints[group_name].items():
            color = JOINT_COLORS.get(joint_name, "black")
            size = 52 if joint_name in {"WRIST", "FOREARM_BASE"} else 38
            ax.scatter(point[0], point[1], point[2], color=color, s=size, depthshade=True)
            if joint_name in {"WRIST", "CMC", "MCP", "TIP"}:
                label = joint_name if group_name == "root" else f"{group_name}:{joint_name}"
                ax.text(point[0], point[1], point[2], label, fontsize=7)


# =============================================================================
# 8. Validation and self-test
# =============================================================================


def segment_length(hand_joints: Mapping[str, Mapping[str, np.ndarray]], group: str, a: str, b: str) -> float:
    return float(np.linalg.norm(get_point(hand_joints, group, a) - get_point(hand_joints, group, b)))


def validate_bone_lengths(
    hand_joints: Mapping[str, Mapping[str, np.ndarray]],
    *,
    tol: float = 1e-6,
    verbose: bool = False,
) -> bool:
    """Verify that computed segment lengths match configuration."""
    ok = True

    checks = [
        ("thumb", "CMC", "MCP", BONE_LENGTHS["thumb"]["CMC_MCP"]),
        ("thumb", "MCP", "IP", BONE_LENGTHS["thumb"]["MCP_IP"]),
        ("thumb", "IP", "TIP", BONE_LENGTHS["thumb"]["IP_TIP"]),
    ]
    for finger_name in FOUR_FINGER_NAMES:
        checks.extend(
            [
                (finger_name, "MCP", "PIP", BONE_LENGTHS[finger_name]["MCP_PIP"]),
                (finger_name, "PIP", "DIP", BONE_LENGTHS[finger_name]["PIP_DIP"]),
                (finger_name, "DIP", "TIP", BONE_LENGTHS[finger_name]["DIP_TIP"]),
            ]
        )

    for group, a, b, expected in checks:
        measured = segment_length(hand_joints, group, a, b)
        if abs(measured - expected) > tol:
            ok = False
            if verbose:
                print(
                    f"[bone-length] {group}.{a}->{b}: measured {measured:.8f}, "
                    f"expected {expected:.8f}, diff {measured - expected:+.3e}"
                )
    return ok


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = normalize(v1)
    v2n = normalize(v2)
    return float(np.rad2deg(math.acos(float(np.clip(np.dot(v1n, v2n), -1.0, 1.0)))))


def validate_open_pose(
    hand_joints: Mapping[str, Mapping[str, np.ndarray]],
    *,
    tol_z: float = 1e-6,
    verbose: bool = False,
) -> bool:
    """Check key assumptions of the default open-palm pose."""
    ok = True
    _, points = flatten_joints(hand_joints)
    max_abs_z = float(np.max(np.abs(points[:, 2])))
    if max_abs_z > tol_z:
        ok = False
        if verbose:
            print(f"[open-pose] max |z|={max_abs_z:.8f} > {tol_z}")

    wrist = get_point(hand_joints, "root", "WRIST")
    middle_mcp = get_point(hand_joints, "middle", "MCP")
    wrist_to_middle_mcp = float(np.linalg.norm(middle_mcp - wrist))
    if wrist_to_middle_mcp < 35.0:
        ok = False
        if verbose:
            print(f"[open-pose] wrist too close to middle MCP: {wrist_to_middle_mcp:.3f}")

    # Four fingers should be nearly parallel in open pose. The small MCP spread
    # means this is approximate, not exact.
    ref = get_point(hand_joints, "middle", "PIP") - get_point(hand_joints, "middle", "MCP")
    for finger_name in FOUR_FINGER_NAMES:
        v = get_point(hand_joints, finger_name, "PIP") - get_point(hand_joints, finger_name, "MCP")
        angle = angle_between(ref, v)
        if angle > 8.0:
            ok = False
            if verbose:
                print(f"[open-pose] {finger_name} deviates from middle by {angle:.3f} deg")

    thumb_vec = get_point(hand_joints, "thumb", "MCP") - get_point(hand_joints, "thumb", "CMC")
    thumb_xy = np.array([thumb_vec[0], thumb_vec[1], 0.0], dtype=float)
    thumb_angle = angle_between(thumb_xy, np.array([0.0, 1.0, 0.0], dtype=float))
    if abs(thumb_angle - 30.0) > 2.0:
        ok = False
        if verbose:
            print(f"[open-pose] thumb angle {thumb_angle:.3f} deg, expected about 30 deg")

    return ok


def run_self_test() -> bool:
    """Run basic deterministic checks without opening an animation window."""
    print("Running dexterous_hand_sim self-test...")
    all_ok = True

    open_angles = pose_open()
    open_joints = get_hand_joints(open_angles)
    checks = [
        ("open angle limits", validate_angle_limits(open_angles, verbose=True)),
        ("open bone lengths", validate_bone_lengths(open_joints, verbose=True)),
        ("open pose geometry", validate_open_pose(open_joints, verbose=True)),
    ]

    for pose_name, pose_fn in POSE_LIBRARY.items():
        angles = pose_fn()
        joints = get_hand_joints(angles)
        checks.append((f"{pose_name} angle limits", validate_angle_limits(angles, verbose=True)))
        checks.append((f"{pose_name} bone lengths", validate_bone_lengths(joints, verbose=True)))

    for pose_name, pose_fn in PHASE_POSE_LIBRARY.items():
        for phase in (0.0, 0.25, 0.5, 0.75):
            angles = pose_fn(phase)
            joints = get_hand_joints(angles)
            checks.append((f"{pose_name} phase={phase:.2f} angle limits", validate_angle_limits(angles, verbose=True)))
            checks.append((f"{pose_name} phase={phase:.2f} bone lengths", validate_bone_lengths(joints, verbose=True)))

    for name, ok in checks:
        print(f"  {'OK' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok

    print("Self-test result:", "PASS" if all_ok else "FAIL")
    return all_ok


# =============================================================================
# 9. Interaction and animation loop
# =============================================================================


class AnimationController:
    """State machine for automatic pose cycling and keyboard controls."""

    def __init__(self) -> None:
        self.action_names = list(POSE_LIBRARY.keys()) + list(PHASE_POSE_LIBRARY.keys())
        self.index = 0
        self.paused = False
        self.current_angles = clamp_angles(pose_open())
        self.start_angles = copy.deepcopy(self.current_angles)
        self.target_angles = self._pose_for_current_action(0.0)
        self.transition_frame = TRANSITION_FRAMES
        self.hold_frame = 0
        self.phase_frame = 0
        self.in_transition = False

    @property
    def action_name(self) -> str:
        return self.action_names[self.index]

    @property
    def is_phase_action(self) -> bool:
        return self.action_name in PHASE_POSE_LIBRARY

    def _pose_for_current_action(self, phase: float = 0.0) -> AngleDict:
        name = self.action_name
        if name in POSE_LIBRARY:
            return clamp_angles(POSE_LIBRARY[name]())
        return clamp_angles(PHASE_POSE_LIBRARY[name](phase))

    def set_action_index(self, new_index: int) -> None:
        self.index = new_index % len(self.action_names)
        self.start_angles = copy.deepcopy(self.current_angles)
        self.target_angles = self._pose_for_current_action(0.0)
        self.transition_frame = 0
        self.hold_frame = 0
        self.phase_frame = 0
        self.in_transition = True

    def reset(self) -> None:
        self.index = 0
        self.paused = False
        self.current_angles = clamp_angles(pose_open())
        self.start_angles = copy.deepcopy(self.current_angles)
        self.target_angles = self._pose_for_current_action(0.0)
        self.transition_frame = TRANSITION_FRAMES
        self.hold_frame = 0
        self.phase_frame = 0
        self.in_transition = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def step(self) -> Tuple[AngleDict, str]:
        if self.paused:
            return self.current_angles, self.action_name

        if self.in_transition:
            t = (self.transition_frame + 1) / float(max(1, TRANSITION_FRAMES))
            self.current_angles = interpolate_angles(self.start_angles, self.target_angles, t)
            self.transition_frame += 1
            if self.transition_frame >= TRANSITION_FRAMES:
                self.in_transition = False
                self.current_angles = copy.deepcopy(self.target_angles)
            return self.current_angles, self.action_name

        if self.is_phase_action:
            phase = (self.phase_frame % PHASE_FRAMES) / float(PHASE_FRAMES)
            self.current_angles = self._pose_for_current_action(phase)
            self.phase_frame += 1
            if self.phase_frame >= PHASE_FRAMES * PHASE_CYCLES_PER_ACTION:
                self.set_action_index(self.index + 1)
            return self.current_angles, self.action_name

        self.current_angles = copy.deepcopy(self.target_angles)
        self.hold_frame += 1
        if self.hold_frame >= HOLD_FRAMES:
            self.set_action_index(self.index + 1)
        return self.current_angles, self.action_name


def run_animation() -> None:
    """Run the interactive Matplotlib animation."""
    plt = get_pyplot()
    fig, ax = init_3d_plot()
    controller = AnimationController()

    def on_key_press(event) -> None:
        key = (event.key or "").lower()
        if key == "n":
            controller.set_action_index(controller.index + 1)
        elif key == "p":
            controller.set_action_index(controller.index - 1)
        elif key == " ":
            controller.toggle_pause()
        elif key == "space":
            controller.toggle_pause()
        elif key == "r":
            controller.reset()
        elif key in {"q", "escape", "esc"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    while plt.fignum_exists(fig.number):
        angles, action_name = controller.step()
        hand_joints = get_hand_joints(angles)
        draw_hand(ax, hand_joints, action_name, paused=controller.paused)
        fig.canvas.draw_idle()
        plt.pause(1.0 / FPS)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dexterous hand kinematic simulation.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run validation checks and exit without opening the animation window.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    run_animation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
