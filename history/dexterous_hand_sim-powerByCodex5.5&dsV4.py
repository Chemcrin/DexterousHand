"""机械灵巧手 3D 运动学仿真。

运行方式：
    python dexterous_hand_sim.py
    python dexterous_hand_sim.py --validate

本文件刻意保持单文件结构，便于直接运行和后续拆分模块。
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401，保留该导入以显式启用 3D 绘图。


# =============================================================================
# 1. 全局配置
# =============================================================================

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
FOUR_FINGER_NAMES = ["index", "middle", "ring", "little"]

FOUR_FINGER_JOINTS = ["MCP", "PIP", "DIP", "TIP"]
THUMB_JOINTS = ["CMC", "MCP", "IP", "TIP"]
ROOT_JOINTS = ["FOREARM_BASE", "WRIST"]

LOCAL_FORWARD = np.array([0.0, 1.0, 0.0])
WRIST_POSITION = np.array([0.0, 0.0, 0.0])
FOREARM_LENGTH = 60.0
THUMB_DEFAULT_SPREAD_DEG = 30.0

FPS = 30
TRANSITION_FRAMES = 42
HOLD_FRAMES = 30
PHASE_ACTION_FRAMES = 120

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
    "thumb": np.array([-28.0, 18.0, 0.0]),
    "index": np.array([-18.0, 48.0, 0.0]),
    "middle": np.array([-6.0, 52.0, 0.0]),
    "ring": np.array([7.0, 49.0, 0.0]),
    "little": np.array([19.0, 43.0, 0.0]),
}

FINGER_BASE_DIRECTIONS = {
    "thumb": np.array([-0.5, 0.8660254, 0.0]),
    "index": np.array([0.0, 1.0, 0.0]),
    "middle": np.array([0.0, 1.0, 0.0]),
    "ring": np.array([0.0, 1.0, 0.0]),
    "little": np.array([0.0, 1.0, 0.0]),
}

JOINT_COLORS = {
    "FOREARM_BASE": "0.25",
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
FOREARM_COLOR = "black"

BONE_CONNECTIONS = [
    ("thumb", "CMC", "thumb", "MCP"),
    ("thumb", "MCP", "thumb", "IP"),
    ("thumb", "IP", "thumb", "TIP"),
]

for _finger_name in FOUR_FINGER_NAMES:
    BONE_CONNECTIONS.extend(
        [
            (_finger_name, "MCP", _finger_name, "PIP"),
            (_finger_name, "PIP", _finger_name, "DIP"),
            (_finger_name, "DIP", _finger_name, "TIP"),
        ]
    )

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


# =============================================================================
# 2. 旋转矩阵工具函数
# =============================================================================


def deg_to_rad(angle_deg: float) -> float:
    """角度制转弧度制。"""
    return float(np.deg2rad(angle_deg))


def rot_x(angle_rad: float) -> np.ndarray:
    """绕 X 轴旋转。正角度使局部 +Y 朝 +Z 弯曲。"""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rot_y(angle_rad: float) -> np.ndarray:
    """绕 Y 轴旋转。"""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rot_z(angle_rad: float) -> np.ndarray:
    """绕 Z 轴旋转。"""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """安全归一化。"""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        if fallback is None:
            return np.zeros_like(vector, dtype=float)
        return fallback.astype(float).copy()
    return vector.astype(float) / norm


def rot_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """使用 Rodrigues 公式计算任意轴旋转矩阵。"""
    unit_axis = normalize(axis, fallback=np.array([0.0, 1.0, 0.0]))
    x, y, z = unit_axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def smoothstep(t: float) -> float:
    """平滑插值系数，减少动作起止时的突兀感。"""
    clipped = float(np.clip(t, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


# =============================================================================
# 3. 默认角度、限位和插值
# =============================================================================


def neutral_angles() -> dict:
    """返回默认张手姿态。"""
    angle_dict = {
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
    }

    for finger_name in FOUR_FINGER_NAMES:
        angle_dict[finger_name] = {
            "mcp_abd_add": 0.0,
            "mcp_flex": 0.0,
            "pip_flex": 0.0,
            "dip_flex": 0.0,
        }

    return angle_dict


def clamp_value(value: float, limits: tuple[float, float]) -> float:
    return float(np.clip(value, limits[0], limits[1]))


def clamp_angles(angle_dict: dict) -> dict:
    """根据关节活动范围限制角度。"""
    result = copy.deepcopy(angle_dict)

    for key, limits in WRIST_LIMITS.items():
        result["root"][key] = clamp_value(result["root"][key], limits)

    for key, limits in THUMB_LIMITS.items():
        result["thumb"][key] = clamp_value(result["thumb"][key], limits)

    for finger_name in FOUR_FINGER_NAMES:
        for key, limits in FOUR_FINGER_LIMITS.items():
            result[finger_name][key] = clamp_value(result[finger_name][key], limits)

    return result


def interpolate_angles(start_angles: dict, end_angles: dict, t: float) -> dict:
    """递归插值两个角度字典。"""
    result = {}
    for key in start_angles:
        start_value = start_angles[key]
        end_value = end_angles[key]
        if isinstance(start_value, dict):
            result[key] = interpolate_angles(start_value, end_value, t)
        else:
            result[key] = float(start_value + (end_value - start_value) * t)
    return result


def set_four_finger_flex(
    angle_dict: dict,
    finger_name: str,
    mcp_flex: float,
    pip_flex: float,
    dip_flex: float,
    mcp_abd_add: float | None = None,
) -> None:
    """快速设置四指中某根手指的屈曲角。"""
    angle_dict[finger_name]["mcp_flex"] = mcp_flex
    angle_dict[finger_name]["pip_flex"] = pip_flex
    angle_dict[finger_name]["dip_flex"] = dip_flex
    if mcp_abd_add is not None:
        angle_dict[finger_name]["mcp_abd_add"] = mcp_abd_add


def curl_four_fingers(
    angle_dict: dict,
    mcp_flex: float,
    pip_flex: float,
    dip_flex: float,
    exclude: set[str] | None = None,
) -> None:
    """批量弯曲四指，可排除保持伸直的手指。"""
    excluded = exclude or set()
    for finger_name in FOUR_FINGER_NAMES:
        if finger_name not in excluded:
            set_four_finger_flex(angle_dict, finger_name, mcp_flex, pip_flex, dip_flex)


def fold_thumb(angle_dict: dict, strength: float = 1.0) -> None:
    """让拇指自然内收，用于数字手势或握拳。"""
    strength = float(np.clip(strength, 0.0, 1.0))
    thumb = angle_dict["thumb"]
    thumb["cmc_abd_add"] = 12.0 * strength
    thumb["cmc_flex"] = 32.0 * strength
    thumb["cmc_axial_rot"] = 42.0 * strength
    thumb["mcp_flex"] = 34.0 * strength
    thumb["ip_flex"] = 30.0 * strength


# =============================================================================
# 4. 正向运动学
# =============================================================================


def get_root_transform(root_angles: dict) -> np.ndarray:
    """计算整只手相对腕部的根姿态。"""
    flex = deg_to_rad(root_angles["wrist_flex_ext"])
    radial = deg_to_rad(root_angles["wrist_radial_ulnar"])
    prono = deg_to_rad(root_angles["forearm_prono_supination"])
    return rot_y(prono) @ rot_z(radial) @ rot_x(flex)


def compute_four_finger_joints(
    finger_name: str,
    finger_angles: dict,
    root_R: np.ndarray,
) -> dict:
    """计算四指中某一根手指的 MCP/PIP/DIP/TIP 坐标。"""
    lengths = BONE_LENGTHS[finger_name]

    mcp_abd = deg_to_rad(finger_angles["mcp_abd_add"])
    mcp_flex = deg_to_rad(finger_angles["mcp_flex"])
    pip_flex = deg_to_rad(finger_angles["pip_flex"])
    dip_flex = deg_to_rad(finger_angles["dip_flex"])

    mcp = root_R @ PALM_BASE_POINTS[finger_name]

    base_R = root_R @ rot_z(mcp_abd)
    proximal_R = base_R @ rot_x(mcp_flex)
    middle_R = proximal_R @ rot_x(pip_flex)
    distal_R = middle_R @ rot_x(dip_flex)

    pip = mcp + (proximal_R @ LOCAL_FORWARD) * lengths["MCP_PIP"]
    dip = pip + (middle_R @ LOCAL_FORWARD) * lengths["PIP_DIP"]
    tip = dip + (distal_R @ LOCAL_FORWARD) * lengths["DIP_TIP"]

    return {
        "MCP": mcp,
        "PIP": pip,
        "DIP": dip,
        "TIP": tip,
    }


def compute_thumb_joints(thumb_angles: dict, root_R: np.ndarray) -> dict:
    """计算拇指 CMC/MCP/IP/TIP 坐标。"""
    lengths = BONE_LENGTHS["thumb"]

    cmc_abd = deg_to_rad(thumb_angles["cmc_abd_add"])
    cmc_flex = deg_to_rad(thumb_angles["cmc_flex"])
    cmc_axial = deg_to_rad(thumb_angles["cmc_axial_rot"])
    mcp_abd = deg_to_rad(thumb_angles["mcp_abd_add"])
    mcp_flex = deg_to_rad(thumb_angles["mcp_flex"])
    ip_flex = deg_to_rad(thumb_angles["ip_flex"])

    cmc = root_R @ PALM_BASE_POINTS["thumb"]

    thumb_spread = deg_to_rad(THUMB_DEFAULT_SPREAD_DEG)
    cmc_R = root_R @ rot_z(thumb_spread) @ rot_z(cmc_abd) @ rot_x(cmc_flex)

    # 轴向旋转围绕当前拇指掌骨方向，使用 Rodrigues 公式。
    thumb_axis = cmc_R @ LOCAL_FORWARD
    cmc_R = cmc_R @ rot_axis(thumb_axis, cmc_axial)

    mcp = cmc + (cmc_R @ LOCAL_FORWARD) * lengths["CMC_MCP"]

    mcp_R = cmc_R @ rot_z(mcp_abd) @ rot_x(mcp_flex)
    ip = mcp + (mcp_R @ LOCAL_FORWARD) * lengths["MCP_IP"]

    ip_R = mcp_R @ rot_x(ip_flex)
    tip = ip + (ip_R @ LOCAL_FORWARD) * lengths["IP_TIP"]

    return {
        "CMC": cmc,
        "MCP": mcp,
        "IP": ip,
        "TIP": tip,
    }


def get_hand_joints(angle_dict: dict) -> dict:
    """根据完整角度字典计算整只手的三维关节点。"""
    safe_angles = clamp_angles(angle_dict)
    root_R = get_root_transform(safe_angles["root"])

    hand_joints = {
        "root": {
            "FOREARM_BASE": root_R @ np.array([0.0, -FOREARM_LENGTH, 0.0]),
            "WRIST": WRIST_POSITION.copy(),
        },
        "thumb": compute_thumb_joints(safe_angles["thumb"], root_R),
    }

    for finger_name in FOUR_FINGER_NAMES:
        hand_joints[finger_name] = compute_four_finger_joints(
            finger_name,
            safe_angles[finger_name],
            root_R,
        )

    return hand_joints


# =============================================================================
# 5. 动作定义
# =============================================================================


def pose_open() -> dict:
    """摊开手掌：四指平行伸直，拇指自然张开。"""
    return neutral_angles()


def pose_ok() -> dict:
    """OK 手势：拇指和食指视觉上靠近，其余手指自然弯曲。"""
    angles = neutral_angles()
    thumb = angles["thumb"]
    thumb["cmc_abd_add"] = -6.0
    thumb["cmc_flex"] = 28.0
    thumb["cmc_axial_rot"] = 42.0
    thumb["mcp_flex"] = 24.0
    thumb["ip_flex"] = 28.0

    set_four_finger_flex(angles, "index", 38.0, 52.0, 32.0, mcp_abd_add=7.0)
    set_four_finger_flex(angles, "middle", 22.0, 30.0, 18.0)
    set_four_finger_flex(angles, "ring", 26.0, 36.0, 22.0)
    set_four_finger_flex(angles, "little", 30.0, 42.0, 25.0)
    return angles


def pose_precision_grasp() -> dict:
    """精准抓握：拇指对掌，食指和中指轻曲配合。"""
    angles = neutral_angles()
    thumb = angles["thumb"]
    thumb["cmc_abd_add"] = 15.0
    thumb["cmc_flex"] = 32.0
    thumb["cmc_axial_rot"] = 48.0
    thumb["mcp_flex"] = 26.0
    thumb["ip_flex"] = 18.0

    set_four_finger_flex(angles, "index", 28.0, 34.0, 20.0, mcp_abd_add=4.0)
    set_four_finger_flex(angles, "middle", 24.0, 30.0, 18.0)
    set_four_finger_flex(angles, "ring", 16.0, 20.0, 12.0)
    set_four_finger_flex(angles, "little", 14.0, 18.0, 10.0)
    return angles


def pose_fist() -> dict:
    """握拳：四指大角度屈曲，拇指自然压向掌心。"""
    angles = neutral_angles()
    set_four_finger_flex(angles, "index", 78.0, 98.0, 62.0)
    set_four_finger_flex(angles, "middle", 82.0, 105.0, 68.0)
    set_four_finger_flex(angles, "ring", 80.0, 102.0, 66.0)
    set_four_finger_flex(angles, "little", 75.0, 96.0, 60.0)
    fold_thumb(angles, strength=1.0)
    return angles


def pose_thumb_up() -> dict:
    """竖大拇指：四指握拳，拇指保持伸展。"""
    angles = pose_fist()
    thumb = angles["thumb"]
    thumb["cmc_abd_add"] = 20.0
    thumb["cmc_flex"] = -5.0
    thumb["cmc_axial_rot"] = 0.0
    thumb["mcp_abd_add"] = 0.0
    thumb["mcp_flex"] = 0.0
    thumb["ip_flex"] = 0.0
    angles["root"]["forearm_prono_supination"] = 0.0
    angles["root"]["wrist_radial_ulnar"] = 10.0
    return angles


def pose_number_1() -> dict:
    """数字 1：食指伸直，其余手指弯曲。"""
    angles = neutral_angles()
    curl_four_fingers(angles, 72.0, 92.0, 58.0, exclude={"index"})
    fold_thumb(angles, strength=0.9)
    return angles


def pose_number_2() -> dict:
    """数字 2：食指和中指伸直，其余手指弯曲。"""
    angles = neutral_angles()
    curl_four_fingers(angles, 72.0, 92.0, 58.0, exclude={"index", "middle"})
    fold_thumb(angles, strength=0.9)
    return angles


def pose_number_3() -> dict:
    """数字 3：食指、中指、无名指伸直，其余手指弯曲。"""
    angles = neutral_angles()
    curl_four_fingers(angles, 72.0, 92.0, 58.0, exclude={"index", "middle", "ring"})
    fold_thumb(angles, strength=0.9)
    return angles


def pose_number_5() -> dict:
    """数字 5：全手张开。"""
    return pose_open()


def pose_little_up() -> dict:
    """竖小拇指：小指伸直，其余手指弯曲。"""
    angles = neutral_angles()
    curl_four_fingers(angles, 72.0, 92.0, 58.0, exclude={"little"})
    fold_thumb(angles, strength=0.9)
    return angles


def pose_half_fist() -> dict:
    """半握拳：所有手指中度弯曲。"""
    angles = neutral_angles()
    set_four_finger_flex(angles, "index", 38.0, 48.0, 28.0)
    set_four_finger_flex(angles, "middle", 42.0, 54.0, 32.0)
    set_four_finger_flex(angles, "ring", 40.0, 50.0, 30.0)
    set_four_finger_flex(angles, "little", 36.0, 46.0, 28.0)

    thumb = angles["thumb"]
    thumb["cmc_abd_add"] = 8.0
    thumb["cmc_flex"] = 20.0
    thumb["cmc_axial_rot"] = 26.0
    thumb["mcp_flex"] = 18.0
    thumb["ip_flex"] = 12.0
    return angles


def pose_finger_wave_phase(phase: float) -> dict:
    """手指依次屈伸：四指以相位差依次弯曲和伸直。"""
    angles = neutral_angles()
    offsets = {
        "index": 0.00,
        "middle": 0.18,
        "ring": 0.36,
        "little": 0.54,
    }

    for finger_name, offset in offsets.items():
        local_phase = (phase - offset) % 1.0
        factor = 0.5 - 0.5 * math.cos(2.0 * math.pi * local_phase)
        set_four_finger_flex(
            angles,
            finger_name,
            62.0 * factor,
            82.0 * factor,
            48.0 * factor,
        )

    return angles


def pose_thumb_opposition_phase(phase: float) -> dict:
    """拇指对掌运动：拇指从外侧转向掌心，再回到张开姿态。"""
    angles = neutral_angles()
    factor = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)

    thumb = angles["thumb"]
    thumb["cmc_abd_add"] = 5.0 + 25.0 * factor
    thumb["cmc_flex"] = 40.0 * factor
    thumb["cmc_axial_rot"] = 55.0 * factor
    thumb["mcp_flex"] = 35.0 * factor
    thumb["ip_flex"] = 35.0 * factor

    for name in FOUR_FINGER_NAMES:
        set_four_finger_flex(angles, name, 25.0 * factor, 30.0 * factor, 20.0 * factor)
    return angles


def pose_wave_phase(phase: float) -> dict:
    """挥手动作：由腕部和前臂姿态驱动，手指保持张开并轻微摆动。"""
    angles = neutral_angles()
    wave = math.sin(2.0 * math.pi * phase)
    subtle = math.sin(4.0 * math.pi * phase)

    angles["root"]["wrist_radial_ulnar"] = 20.0 * wave
    angles["root"]["forearm_prono_supination"] = 18.0 * math.sin(2.0 * math.pi * phase + math.pi / 5.0)
    angles["root"]["wrist_flex_ext"] = 8.0 * subtle

    angles["index"]["mcp_abd_add"] = -3.0 * wave
    angles["middle"]["mcp_abd_add"] = -1.0 * wave
    angles["ring"]["mcp_abd_add"] = 2.0 * wave
    angles["little"]["mcp_abd_add"] = 4.0 * wave
    return angles


PoseFunc = Callable[[], dict]
PhasePoseFunc = Callable[[float], dict]

ACTION_SEQUENCE: list[dict] = [
    {"name": "摊开手掌", "func": pose_open, "phase": False},
    {"name": "OK手势", "func": pose_ok, "phase": False},
    {"name": "精准抓握", "func": pose_precision_grasp, "phase": False},
    {"name": "握拳", "func": pose_fist, "phase": False},
    {"name": "竖大拇指", "func": pose_thumb_up, "phase": False},
    {"name": "数字1", "func": pose_number_1, "phase": False},
    {"name": "数字2", "func": pose_number_2, "phase": False},
    {"name": "数字3", "func": pose_number_3, "phase": False},
    {"name": "数字5", "func": pose_number_5, "phase": False},
    {"name": "竖小拇指", "func": pose_little_up, "phase": False},
    {"name": "半握拳", "func": pose_half_fist, "phase": False},
    {"name": "手指依次屈伸", "func": pose_finger_wave_phase, "phase": True},
    {"name": "拇指对掌运动", "func": pose_thumb_opposition_phase, "phase": True},
    {"name": "挥手动作", "func": pose_wave_phase, "phase": True},
]


# =============================================================================
# 6. 渲染函数
# =============================================================================


def setup_axes(ax, action_name: str = "", paused: bool = False) -> None:
    """统一设置 3D 坐标轴显示。"""
    ax.set_xlim(-90.0, 80.0)
    ax.set_ylim(-75.0, 145.0)
    ax.set_zlim(-35.0, 115.0)
    ax.set_box_aspect((170.0, 220.0, 150.0))
    ax.set_xlabel("X 拇指侧 -> 小指侧")
    ax.set_ylabel("Y 腕部 -> 指尖")
    ax.set_zlabel("Z 掌心方向")
    ax.view_init(elev=24.0, azim=-65.0)
    status = "暂停" if paused else "播放"
    ax.set_title(f"机械灵巧手 3D 仿真 | {action_name} | {status}", pad=14)


def init_3d_plot() -> tuple:
    """初始化 Matplotlib 3D 画布。"""
    fig = plt.figure(figsize=(11.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.1)
    setup_axes(ax, "摊开手掌")
    fig.text(
        0.5,
        0.025,
        "按键：n 下一个 | p 上一个 | Space 暂停/继续 | r 重置 | q/Esc 退出",
        ha="center",
        va="center",
        fontsize=10,
    )
    return fig, ax


def get_joint_point(hand_joints: dict, group_name: str, joint_name: str) -> np.ndarray:
    return hand_joints[group_name][joint_name]


def draw_connection(
    ax,
    hand_joints: dict,
    connection: tuple[str, str, str, str],
    color: str,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    group_a, joint_a, group_b, joint_b = connection
    p0 = get_joint_point(hand_joints, group_a, joint_a)
    p1 = get_joint_point(hand_joints, group_b, joint_b)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def draw_hand(ax, hand_joints: dict, action_name: str = "", paused: bool = False) -> None:
    """绘制整只手。第一版采用清屏重绘，逻辑更直观。"""
    ax.cla()
    setup_axes(ax, action_name, paused)

    draw_connection(
        ax,
        hand_joints,
        ("root", "FOREARM_BASE", "root", "WRIST"),
        FOREARM_COLOR,
        linewidth=4.0,
    )

    for connection in PALM_CONNECTIONS:
        draw_connection(ax, hand_joints, connection, PALM_BONE_COLOR, linewidth=1.4, alpha=0.65)

    for connection in BONE_CONNECTIONS:
        draw_connection(ax, hand_joints, connection, BONE_COLOR, linewidth=3.0, alpha=0.95)

    grouped_points: dict[str, list[np.ndarray]] = {}
    for group_points in hand_joints.values():
        for joint_name, point in group_points.items():
            grouped_points.setdefault(joint_name, []).append(point)

    for joint_name, points in grouped_points.items():
        stacked = np.vstack(points)
        ax.scatter(
            stacked[:, 0],
            stacked[:, 1],
            stacked[:, 2],
            s=58 if joint_name != "TIP" else 44,
            color=JOINT_COLORS.get(joint_name, "black"),
            depthshade=True,
            label=joint_name,
        )

    # 图例只显示关节层级，不对每个点重复标注，避免遮挡手部结构。
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)


# =============================================================================
# 7. 动画和交互
# =============================================================================


class HandAnimationController:
    """管理动作切换、插值、暂停和键盘交互。"""

    def __init__(self, fig, ax) -> None:
        self.fig = fig
        self.ax = ax
        self.action_index = 0
        self.frame_in_action = 0
        self.paused = False
        self.current_angles = pose_open()
        self.start_angles = copy.deepcopy(self.current_angles)
        self.animation = None

    def current_spec(self) -> dict:
        return ACTION_SEQUENCE[self.action_index]

    def current_name(self) -> str:
        return str(self.current_spec()["name"])

    def target_for_current_action(self, phase: float = 0.0) -> dict:
        spec = self.current_spec()
        if spec["phase"]:
            func = spec["func"]
            return func(phase)
        func = spec["func"]
        return func()

    def goto_action(self, action_index: int) -> None:
        self.action_index = action_index % len(ACTION_SEQUENCE)
        self.start_angles = copy.deepcopy(self.current_angles)
        self.frame_in_action = 0

    def next_action(self) -> None:
        self.goto_action(self.action_index + 1)

    def previous_action(self) -> None:
        self.goto_action(self.action_index - 1)

    def reset(self) -> None:
        self.action_index = 0
        self.frame_in_action = 0
        self.current_angles = pose_open()
        self.start_angles = copy.deepcopy(self.current_angles)

    def compute_frame_angles(self) -> dict:
        spec = self.current_spec()
        is_phase_action = bool(spec["phase"])

        if is_phase_action:
            if self.frame_in_action < TRANSITION_FRAMES:
                t = smoothstep(self.frame_in_action / max(1, TRANSITION_FRAMES - 1))
                target = self.target_for_current_action(0.0)
                return interpolate_angles(self.start_angles, target, t)

            phase_frame = self.frame_in_action - TRANSITION_FRAMES
            phase = (phase_frame % PHASE_ACTION_FRAMES) / PHASE_ACTION_FRAMES
            return self.target_for_current_action(phase)

        target = self.target_for_current_action(0.0)
        if self.frame_in_action < TRANSITION_FRAMES:
            t = smoothstep(self.frame_in_action / max(1, TRANSITION_FRAMES - 1))
            return interpolate_angles(self.start_angles, target, t)
        return target

    def should_advance(self) -> bool:
        spec = self.current_spec()
        if spec["phase"]:
            return self.frame_in_action >= TRANSITION_FRAMES + PHASE_ACTION_FRAMES
        return self.frame_in_action >= TRANSITION_FRAMES + HOLD_FRAMES

    def update(self, _frame: int):
        if not self.paused:
            self.current_angles = clamp_angles(self.compute_frame_angles())
            self.frame_in_action += 1
            if self.should_advance():
                self.next_action()

        hand_joints = get_hand_joints(self.current_angles)
        draw_hand(self.ax, hand_joints, self.current_name(), self.paused)
        return []

    def on_key_press(self, event) -> None:
        key = event.key
        if key in (" ", "space"):
            self.paused = not self.paused
        elif key == "n":
            self.next_action()
        elif key == "p":
            self.previous_action()
        elif key == "r":
            self.reset()
        elif key in ("q", "escape"):
            plt.close(self.fig)


def run_animation() -> None:
    fig, ax = init_3d_plot()
    controller = HandAnimationController(fig, ax)
    fig.canvas.mpl_connect("key_press_event", controller.on_key_press)
    controller.animation = FuncAnimation(
        fig,
        controller.update,
        interval=1000 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()


# =============================================================================
# 8. 验证与调试
# =============================================================================


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def validate_bone_lengths(hand_joints: dict, tolerance: float = 1e-6) -> tuple[bool, list[str]]:
    """检查所有显式骨段长度是否保持为配置值。"""
    messages = []
    ok = True

    thumb_pairs = [
        ("CMC", "MCP", "CMC_MCP"),
        ("MCP", "IP", "MCP_IP"),
        ("IP", "TIP", "IP_TIP"),
    ]
    for joint_a, joint_b, length_key in thumb_pairs:
        actual = distance(hand_joints["thumb"][joint_a], hand_joints["thumb"][joint_b])
        expected = BONE_LENGTHS["thumb"][length_key]
        if abs(actual - expected) > tolerance:
            ok = False
            messages.append(f"拇指 {joint_a}->{joint_b} 长度异常：{actual:.6f} != {expected:.6f}")

    four_pairs = [
        ("MCP", "PIP", "MCP_PIP"),
        ("PIP", "DIP", "PIP_DIP"),
        ("DIP", "TIP", "DIP_TIP"),
    ]
    for finger_name in FOUR_FINGER_NAMES:
        for joint_a, joint_b, length_key in four_pairs:
            actual = distance(hand_joints[finger_name][joint_a], hand_joints[finger_name][joint_b])
            expected = BONE_LENGTHS[finger_name][length_key]
            if abs(actual - expected) > tolerance:
                ok = False
                messages.append(
                    f"{finger_name} {joint_a}->{joint_b} 长度异常：{actual:.6f} != {expected:.6f}"
                )

    if ok:
        messages.append("骨段长度检查通过。")
    return ok, messages


def angle_between(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = normalize(vector_a)
    unit_b = normalize(vector_b)
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def validate_open_pose(hand_joints: dict, tolerance: float = 1e-6) -> tuple[bool, list[str]]:
    """检查默认张手姿态是否符合讨论后的基线。"""
    messages = []
    ok = True

    all_points = []
    for group_points in hand_joints.values():
        all_points.extend(group_points.values())
    max_abs_z = max(abs(float(point[2])) for point in all_points)
    if max_abs_z > tolerance:
        ok = False
        messages.append(f"默认张手姿态未处于同一平面，最大 |z| = {max_abs_z:.6f}")

    for finger_name in FOUR_FINGER_NAMES:
        direction = hand_joints[finger_name]["PIP"] - hand_joints[finger_name]["MCP"]
        deviation = angle_between(direction, LOCAL_FORWARD)
        if deviation > 1e-5:
            ok = False
            messages.append(f"{finger_name} 默认方向未沿 +Y，偏差 {deviation:.6f} 度")

    thumb_direction = hand_joints["thumb"]["MCP"] - hand_joints["thumb"]["CMC"]
    thumb_angle = angle_between(thumb_direction, LOCAL_FORWARD)
    if abs(thumb_angle - THUMB_DEFAULT_SPREAD_DEG) > 1e-5:
        ok = False
        messages.append(f"拇指默认张开角异常：{thumb_angle:.6f} 度")

    wrist_to_middle = distance(hand_joints["root"]["WRIST"], hand_joints["middle"]["MCP"])
    if wrist_to_middle < 30.0:
        ok = False
        messages.append(f"中指 MCP 与腕部过近：{wrist_to_middle:.6f}")

    if thumb_direction[0] >= 0.0:
        ok = False
        messages.append("拇指默认方向没有朝拇指侧张开。")

    if ok:
        messages.append("默认张手姿态检查通过。")
    return ok, messages


def validate_angle_limits(angle_dict: dict) -> tuple[bool, list[str]]:
    """检查角度是否处于限位内。"""
    clamped = clamp_angles(angle_dict)
    messages = []
    ok = True

    def walk(path: list[str], lhs, rhs) -> None:
        nonlocal ok
        if isinstance(lhs, dict):
            for key in lhs:
                walk(path + [key], lhs[key], rhs[key])
            return
        if abs(float(lhs) - float(rhs)) > 1e-9:
            ok = False
            messages.append(f"{'.'.join(path)} 超出限位：{lhs:.3f} -> {rhs:.3f}")

    walk([], angle_dict, clamped)
    if ok:
        messages.append("角度限位检查通过。")
    return ok, messages


def run_validation() -> bool:
    """运行无需 GUI 的基础验证。"""
    all_ok = True
    print("开始验证机械灵巧手运动学配置...\n")

    open_angles = pose_open()
    open_joints = get_hand_joints(open_angles)

    checks = [
        validate_angle_limits(open_angles),
        validate_bone_lengths(open_joints),
        validate_open_pose(open_joints),
    ]

    for action in ACTION_SEQUENCE:
        if action["phase"]:
            sample_angles = action["func"](0.35)
        else:
            sample_angles = action["func"]()
        sample_joints = get_hand_joints(sample_angles)
        checks.append(validate_angle_limits(sample_angles))
        checks.append(validate_bone_lengths(sample_joints))

    for ok, messages in checks:
        all_ok = all_ok and ok
        for message in messages:
            print(message)

    print("\n验证结果：" + ("通过" if all_ok else "存在问题"))
    return all_ok


# =============================================================================
# 9. 程序入口
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="机械灵巧手 3D 运动学仿真")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="只运行运动学验证，不打开 Matplotlib 窗口。",
    )
    args = parser.parse_args()

    if args.validate:
        return 0 if run_validation() else 1

    run_animation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
