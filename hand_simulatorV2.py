"""
机械灵巧手仿真 - 单文件完整版
直接运行即可，无需额外模块文件
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict, Callable, Tuple

# ==================== 第一部分：数学引擎 ====================

def rot_x(angle_rad: float) -> np.ndarray:
    """生成绕 X 轴旋转的 3x3 旋转矩阵"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(angle_rad: float) -> np.ndarray: 
    """生成绕 Y 轴旋转的 3x3 旋转矩阵"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(angle_rad: float) -> np.ndarray:
    """生成绕 Z 轴旋转的 3x3 旋转矩阵"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def rot_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """绕任意轴旋转（罗德里格斯公式）"""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    v = 1 - c
    return np.array([
        [c + x*x*v, x*y*v - z*s, x*z*v + y*s],
        [x*y*v + z*s, c + y*y*v, y*z*v - x*s],
        [x*z*v - y*s, y*z*v + x*s, c + z*z*v]
    ], dtype=np.float64)

def translation_matrix(offset: np.ndarray) -> np.ndarray:
    """生成 4x4 平移矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = offset
    return T

def rotation_matrix_to_transform(R: np.ndarray) -> np.ndarray:
    """将 3x3 旋转矩阵嵌入 4x4 齐次矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T

def combine_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """组合旋转和平移为 4x4 变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class KinematicChain:
    """单根手指的运动链"""
    def __init__(self, name: str):
        self.name = name
        self.rest_translations: List[np.ndarray] = []
        self.axes: List[List[np.ndarray]] = []
        self.angles: List[List[float]] = []

    def add_joint(self, offset: np.ndarray, axes: List[np.ndarray], 
                  init_angles: Optional[List[float]] = None):
        self.rest_translations.append(offset.copy())
        self.axes.append(axes)
        if init_angles is None:
            init_angles = [0.0] * len(axes)
        self.angles.append(init_angles)

    def set_joint_angles(self, joint_idx: int, angles: List[float]):
        if joint_idx < len(self.angles):
            self.angles[joint_idx] = angles.copy()

    def compute_transforms(self) -> List[np.ndarray]:
        transforms = [np.eye(4, dtype=np.float64)]
        current_T = np.eye(4, dtype=np.float64)
        for i, offset in enumerate(self.rest_translations):
            T_trans = translation_matrix(offset)
            R_joint = np.eye(3, dtype=np.float64)
            for axis, ang in zip(self.axes[i], self.angles[i]):
                if not np.isclose(ang, 0.0):
                    R_joint = rot_axis(axis, ang) @ R_joint
            T_local = T_trans @ rotation_matrix_to_transform(R_joint)
            current_T = current_T @ T_local
            transforms.append(current_T.copy())
        return transforms

class HandKinematics:
    """整个手部的运动学管理器"""
    def __init__(self):
        self.palm_transform = np.eye(4, dtype=np.float64)
        self.fingers: Dict[str, KinematicChain] = {}

    def add_finger(self, name: str, chain: KinematicChain):
        self.fingers[name] = chain

    def set_palm_transform(self, R: np.ndarray, t: np.ndarray):
        self.palm_transform = combine_transform(R, t)

    def compute_all_transforms(self) -> Dict[str, List[np.ndarray]]:
        all_transforms = {}
        for name, finger in self.fingers.items():
            local_transforms = finger.compute_transforms()
            all_transforms[name] = [self.palm_transform @ T for T in local_transforms]
        return all_transforms

    def get_all_joint_positions(self) -> Dict[str, List[np.ndarray]]:
        positions = {}
        all_transforms = self.compute_all_transforms()
        for name, transforms in all_transforms.items():
            positions[name] = [T[:3, 3] for T in transforms]
        return positions

# ==================== 第二部分：手部模型 ====================

class HandModel:
    """灵巧手几何模型"""
    def __init__(self):
        self.hand_kin = HandKinematics()
        self.palm_width = 8.0
        self.palm_length = 6.0
        self.finger_lengths = {
            'thumb': [3.0, 2.5, 2.0],
            'index': [3.5, 2.8, 2.2],
            'middle': [3.8, 3.0, 2.4],
            'ring': [3.5, 2.8, 2.2],
            'pinky': [2.8, 2.2, 1.8]
        }
        # 手指根部偏移（重新校准）
        self.finger_base_offsets = {
            'thumb':   np.array([-3.5, -0.5, -0.5]),  # 拇指在手掌左下侧
            'index':   np.array([-1.8,  0.5,  0.0]),  # 食指稍前
            'middle':  np.array([ 0.0,  0.8,  0.0]),  # 中指最前
            'ring':    np.array([ 1.5,  0.5,  0.0]),  # 无名指
            'pinky':   np.array([ 2.8,  0.0, -0.3])   # 小指稍后
        }
        self._build_thumb()
        self._build_finger('index')
        self._build_finger('middle')
        self._build_finger('ring')
        self._build_finger('pinky')
        self.joint_names = self._collect_joint_names()

    def _build_thumb(self):
        chain = KinematicChain('thumb')
        axes_cmc = [np.array([0, 0, 1]), np.array([1, 0, 0])]
        axes_mcp = [np.array([1, 0, 0])]
        axes_ip  = [np.array([1, 0, 0])]
        l0, l1, l2 = self.finger_lengths['thumb']
    
        # 拇指初始外展角（绕Z轴负方向旋转，使拇指朝向手掌内侧）
        init_abd_angle = np.radians(-40)
        R_init = rot_z(init_abd_angle)
    
        # 基础偏移
        base_offset = self.finger_base_offsets['thumb']
        chain.add_joint(base_offset, axes_cmc, init_angles=[0.0, 0.0])
    
        # 拇指掌骨方向 (旋转后的Y轴)
        dir0 = R_init @ np.array([0, 1, 0])
        chain.add_joint(dir0 * l0, axes_mcp)
        chain.add_joint(np.array([0, l1, 0]), axes_ip)
        chain.add_joint(np.array([0, l2, 0]), [])
    
        self.hand_kin.add_finger('thumb', chain)

    def _build_finger(self, name: str):
        chain = KinematicChain(name)
        axes_mcp = [np.array([1, 0, 0]), np.array([0, 0, 1])]
        axes_pip = [np.array([1, 0, 0])]
        axes_dip = [np.array([1, 0, 0])]
        l0, l1, l2 = self.finger_lengths[name]
        chain.add_joint(self.finger_base_offsets[name], axes_mcp)
        chain.add_joint(np.array([0, l0, 0]), axes_pip)
        chain.add_joint(np.array([0, l1, 0]), axes_dip)
        chain.add_joint(np.array([0, l2, 0]), [])
        self.hand_kin.add_finger(name, chain)

    def _collect_joint_names(self) -> list:
        names = []
        for finger_name, chain in self.hand_kin.fingers.items():
            for i in range(len(chain.angles)):
                if len(chain.axes[i]) > 0:
                    names.append(f"{finger_name}_{i}")
        return names

    def set_joint_angle(self, finger: str, joint_idx: int, dof_idx: int, angle_rad: float):
        chain = self.hand_kin.fingers[finger]
        angles = chain.angles[joint_idx].copy()
        angles[dof_idx] = angle_rad
        chain.set_joint_angles(joint_idx, angles)

    def reset_pose(self):
        for chain in self.hand_kin.fingers.values():
            for i in range(len(chain.angles)):
                chain.set_joint_angles(i, [0.0] * len(chain.angles[i]))

    def get_all_transforms(self):
        return self.hand_kin.compute_all_transforms()

    def get_all_joint_positions(self):
        return self.hand_kin.get_all_joint_positions()

    def apply_gesture(self, gesture_name: str):
        self.reset_pose()
        if gesture_name == 'thumbs_up':
            for finger in ['index', 'middle', 'ring', 'pinky']:
                self.set_joint_angle(finger, 0, 0, np.radians(80))
                self.set_joint_angle(finger, 1, 0, np.radians(90))
                self.set_joint_angle(finger, 2, 0, np.radians(60))
            self.set_joint_angle('thumb', 2, 0, np.radians(-10))
        elif gesture_name == 'fist':
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                if finger == 'thumb':
                    self.set_joint_angle('thumb', 0, 0, np.radians(50))
                    self.set_joint_angle('thumb', 1, 0, np.radians(40))
                else:
                    self.set_joint_angle(finger, 0, 0, np.radians(85))
                    self.set_joint_angle(finger, 1, 0, np.radians(100))
                    self.set_joint_angle(finger, 2, 0, np.radians(70))
        elif gesture_name == 'ok':
            self.set_joint_angle('thumb', 0, 1, np.radians(40))
            self.set_joint_angle('thumb', 0, 0, np.radians(30))
            self.set_joint_angle('index', 0, 0, np.radians(60))
            self.set_joint_angle('index', 1, 0, np.radians(40))
        elif gesture_name == 'point':
            self.set_joint_angle('index', 0, 0, np.radians(0))
            for finger in ['middle', 'ring', 'pinky']:
                self.set_joint_angle(finger, 0, 0, np.radians(85))
                self.set_joint_angle(finger, 1, 0, np.radians(90))
            self.set_joint_angle('thumb', 0, 0, np.radians(50))

# ==================== 第三部分：控制器 ====================

class HandController:
    """手部姿态控制器"""
    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        self.joint_params = self._extract_joint_params()
        self.sliders: List[Slider] = []
        self.update_callback: Optional[Callable[[], None]] = None

    def _extract_joint_params(self) -> List[Tuple[str, int, int, float, np.ndarray]]:
        params = []
        for finger_name, chain in self.model.hand_kin.fingers.items():
            for joint_idx in range(len(chain.angles)):
                axes = chain.axes[joint_idx]
                angles = chain.angles[joint_idx]
                for dof_idx, (axis, angle) in enumerate(zip(axes, angles)):
                    params.append((finger_name, joint_idx, dof_idx, angle, axis.copy()))
        return params

    def set_update_callback(self, callback: Callable[[], None]):
        self.update_callback = callback

    def set_joint_angle(self, finger: str, joint_idx: int, dof_idx: int, angle_rad: float):
        self.model.set_joint_angle(finger, joint_idx, dof_idx, angle_rad)
        for i, (f, j, d, _, _) in enumerate(self.joint_params):
            if f == finger and j == joint_idx and d == dof_idx:
                self.joint_params[i] = (f, j, d, angle_rad, self.joint_params[i][4])
                break
        if self.update_callback:
            self.update_callback()

    def apply_gesture(self, gesture_name: str):
        self.model.apply_gesture(gesture_name)
        for i, (f, j, d, _, axis) in enumerate(self.joint_params):
            new_angle = self.model.hand_kin.fingers[f].angles[j][d]
            self.joint_params[i] = (f, j, d, new_angle, axis)
        self._sync_sliders()
        if self.update_callback:
            self.update_callback()

    def _sync_sliders(self):
        for slider, (_, _, _, angle, _) in zip(self.sliders, self.joint_params):
            slider.set_val(np.degrees(angle))

    def create_slider_callback(self, finger: str, joint_idx: int, dof_idx: int):
        def callback(val_deg: float):
            self.set_joint_angle(finger, joint_idx, dof_idx, np.radians(val_deg))
        return callback

    def get_slider_label(self, finger: str, joint_idx: int, dof_idx: int) -> str:
        joint_names = {0: 'CMC', 1: 'MCP', 2: 'IP'} if finger == 'thumb' else {0: 'MCP', 1: 'PIP', 2: 'DIP'}
        dof_names = {0: 'flex', 1: 'abd'}
        return f"{finger}_{joint_names.get(joint_idx, f'J{joint_idx}')}_{dof_names.get(dof_idx, f'dof{dof_idx}')}"

    def get_joint_count(self) -> int:
        return len(self.joint_params)

def setup_interactive_controls(hand_model, fig, update_view_callback):
    controller = HandController(hand_model)
    controller.set_update_callback(update_view_callback)
    
    num_sliders = controller.get_joint_count()
    n_cols = 4
    n_rows = (num_sliders + n_cols - 1) // n_cols
    
    # 根据滑块行数动态调整底部留白
    slider_height = 0.03
    slider_spacing = 0.02
    total_slider_height = n_rows * (slider_height + slider_spacing)
    bottom_margin = 0.25 + total_slider_height  # 为滑块预留的总高度
    
    plt.subplots_adjust(bottom=bottom_margin)
    
    slider_width = 0.18
    left_margin = 0.08
    h_space = 0.02
    
    # 滑块放置位置（从下往上数）
    for i, (finger, joint_idx, dof_idx, angle_rad, _) in enumerate(controller.joint_params):
        row = i // n_cols
        col = i % n_cols
        ax_slider = fig.add_axes([
            left_margin + col * (slider_width + h_space),
            0.20 + (n_rows - 1 - row) * (slider_height + slider_spacing),  # 从底部向上堆叠
            slider_width,
            slider_height
        ])
        label = controller.get_slider_label(finger, joint_idx, dof_idx)
        slider = Slider(ax_slider, label, -90.0, 90.0, 
                        valinit=np.degrees(angle_rad), valstep=1.0)
        slider.on_changed(controller.create_slider_callback(finger, joint_idx, dof_idx))
        controller.sliders.append(slider)
    
    # 按钮放在滑块下方（靠近窗口底部）
    gestures = ['thumbs_up', 'fist', 'ok', 'point', 'reset']
    button_width = 0.1
    button_height = 0.04
    button_bottom = 0.08  # 离底部距离
    button_start_x = 0.1
    
    for i, gesture in enumerate(gestures):
        ax_btn = fig.add_axes([
            button_start_x + i * (button_width + 0.03),
            button_bottom,
            button_width,
            button_height
        ])
        btn = Button(ax_btn, gesture.replace('_', ' ').title())
        def make_callback(g_name):
            def callback(event):
                if g_name == 'reset':
                    controller.model.reset_pose()
                    controller._sync_sliders()
                else:
                    controller.apply_gesture(g_name)
                update_view_callback()
            return callback
        btn.on_clicked(make_callback(gesture))
    
    return controller

# ==================== 第四部分：可视化 ====================

class HandVisualizer:
    """3D 可视化器"""
    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        self.fig = None
        self.ax: Optional[Axes3D] = None
        self.joint_scatter = None
        self.bone_lines: Dict[str, List] = {}
        self.palm_lines: List = []
        self.joint_color, self.joint_size = 'red', 30
        self.bone_color_map = {'thumb': 'blue', 'index': 'green', 'middle': 'orange', 
                               'ring': 'purple', 'pinky': 'brown'}
        self.palm_color, self.line_width = 'gray', 2.0

    def setup_figure(self, figsize=(10, 8)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title('机械灵巧手 - 点线模型')
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_xlim([-5, 10])
        self.ax.set_ylim([-2, 18])
        self.ax.set_zlim([-5, 10])
        self.ax.set_box_aspect([1, 1, 1])

    def draw_hand(self):
        if self.ax is None:
            raise RuntimeError("请先调用 setup_figure()")
        positions = self.model.get_all_joint_positions()
        all_points = []
        for finger_name, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            points = np.array(pos_list)
            all_points.append(points)
            segments = [(points[i], points[i+1]) for i in range(len(points)-1)]
            color = self.bone_color_map.get(finger_name, 'black')
            if finger_name not in self.bone_lines:
                self.bone_lines[finger_name] = [
                    self.ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], [s[0][2], s[1][2]],
                                 color=color, linewidth=self.line_width)[0] for s in segments
                ]
            else:
                for i, s in enumerate(segments):
                    if i < len(self.bone_lines[finger_name]):
                        self.bone_lines[finger_name][i].set_data([s[0][0], s[1][0]], [s[0][1], s[1][1]])
                        self.bone_lines[finger_name][i].set_3d_properties([s[0][2], s[1][2]])
        root_positions = {f: pos_list[0] for f, pos_list in positions.items() if pos_list}
        palm_points = [root_positions[f] for f in ['thumb', 'index', 'middle', 'ring', 'pinky'] if f in root_positions]
        if len(palm_points) >= 2:
            palm_segments = [(palm_points[i], palm_points[i+1]) for i in range(len(palm_points)-1)]
            if not self.palm_lines:
                self.palm_lines = [
                    self.ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], [s[0][2], s[1][2]],
                                 color=self.palm_color, linewidth=self.line_width, linestyle='--')[0]
                    for s in palm_segments
                ]
            else:
                for i, s in enumerate(palm_segments):
                    if i < len(self.palm_lines):
                        self.palm_lines[i].set_data([s[0][0], s[1][0]], [s[0][1], s[1][1]])
                        self.palm_lines[i].set_3d_properties([s[0][2], s[1][2]])
        all_points = np.vstack(all_points) if all_points else np.empty((0, 3))
        if self.joint_scatter is None:
            self.joint_scatter = self.ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                                                 c=self.joint_color, s=self.joint_size, depthshade=True)
        else:
            self.joint_scatter._offsets3d = (all_points[:, 0], all_points[:, 1], all_points[:, 2])
        self.fig.canvas.draw_idle()

    def update_view(self):
        self.draw_hand()

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print("正在启动机械灵巧手仿真...")
    hand = HandModel()
    viz = HandVisualizer(hand)
    viz.setup_figure(figsize=(12, 8))
    controller = setup_interactive_controls(hand, viz.fig, viz.update_view)
    viz.draw_hand()
    plt.show()