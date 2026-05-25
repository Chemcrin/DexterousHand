"""
模块一：运动学数学引擎 (kinematics.py)
功能：提供旋转矩阵生成、齐次变换组合以及正向运动学树求解的核心函数。
依赖：NumPy
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# -------------------- 基础旋转矩阵生成 --------------------
def rot_x(angle_rad: float) -> np.ndarray:
    """
    生成绕 X 轴旋转的 3x3 旋转矩阵。
    
    参数:
        angle_rad: 旋转角度（弧度）
    返回:
        3x3 旋转矩阵
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0,  0.0],
        [0.0,   c,   -s],
        [0.0,   s,    c]
    ], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    """
    生成绕 Y 轴旋转的 3x3 旋转矩阵。
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [  c,  0.0,    s],
        [0.0,  1.0,  0.0],
        [ -s,  0.0,    c]
    ], dtype=np.float64)


def rot_z(angle_rad: float) -> np.ndarray:
    """
    生成绕 Z 轴旋转的 3x3 旋转矩阵。
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [  c,   -s,  0.0],
        [  s,    c,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64)


def rot_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    生成绕任意单位轴旋转的 3x3 旋转矩阵（罗德里格斯公式）。
    
    参数:
        axis: 单位旋转轴向量，形状 (3,)
        angle_rad: 旋转角度（弧度）
    返回:
        3x3 旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)  # 确保单位化
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    v = 1 - c

    # 反对称矩阵的平方及罗德里格斯公式实现
    return np.array([
        [c + x*x*v,     x*y*v - z*s,   x*z*v + y*s],
        [x*y*v + z*s,   c + y*y*v,     y*z*v - x*s],
        [x*z*v - y*s,   y*z*v + x*s,   c + z*z*v]
    ], dtype=np.float64)


# -------------------- 齐次变换矩阵构建 --------------------
def translation_matrix(offset: np.ndarray) -> np.ndarray:
    """
    根据平移向量生成 4x4 齐次平移矩阵。
    
    参数:
        offset: 平移向量，形状 (3,)
    返回:
        4x4 平移矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = offset
    return T


def rotation_matrix_to_transform(R: np.ndarray) -> np.ndarray:
    """
    将 3x3 旋转矩阵嵌入到 4x4 齐次变换矩阵中（平移部分为零）。
    
    参数:
        R: 3x3 旋转矩阵
    返回:
        4x4 变换矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def combine_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵和平移向量组合为一个 4x4 齐次变换矩阵。
    
    参数:
        R: 3x3 旋转矩阵
        t: 平移向量，形状 (3,)
    返回:
        组合后的 4x4 变换矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_inverse(T: np.ndarray) -> np.ndarray:
    """
    计算齐次变换矩阵的逆矩阵（利用旋转部分的正交性加速）。
    
    参数:
        T: 4x4 齐次变换矩阵
    返回:
        逆矩阵
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T  # 旋转矩阵的逆就是其转置
    t_inv = -R_inv @ t
    return combine_transform(R_inv, t_inv)


# -------------------- 正向运动学树求解器 --------------------
class KinematicChain:
    """
    表示一条运动链（例如一根手指），存储各关节的局部变换信息，
    并支持正向运动学计算。
    """
    def __init__(self, name: str):
        self.name = name
        # 存储每个关节的静止偏移（从父关节到该关节的平移）
        self.rest_translations: List[np.ndarray] = []
        # 存储每个关节的旋转轴向量列表（可能多个自由度）
        self.axes: List[List[np.ndarray]] = []
        # 当前各自由度的角度（弧度）
        self.angles: List[List[float]] = []

    def add_joint(self, 
                  offset: np.ndarray, 
                  axes: List[np.ndarray], 
                  init_angles: Optional[List[float]] = None):
        """
        向运动链末端添加一个关节。
        
        参数:
            offset: 从上一关节坐标系原点到本关节坐标系原点的平移向量
            axes: 该关节允许旋转的轴向量列表（单位向量），每个对应一个自由度
            init_angles: 各自由度的初始角度（弧度），默认为 0
        """
        self.rest_translations.append(offset.copy())
        self.axes.append(axes)
        if init_angles is None:
            init_angles = [0.0] * len(axes)
        self.angles.append(init_angles)

    def set_joint_angles(self, joint_idx: int, angles: List[float]):
        """
        设置指定关节的自由度角度。
        """
        if joint_idx < len(self.angles):
            self.angles[joint_idx] = angles.copy()

    def compute_transforms(self) -> List[np.ndarray]:
        """
        计算该运动链中所有关节的全局（相对于基座）齐次变换矩阵。
        
        返回:
            列表，第 i 个元素为第 i 个关节坐标系的全局变换矩阵。
            列表长度 = 关节数 + 1（包含基座，即单位矩阵）。
        """
        transforms = [np.eye(4, dtype=np.float64)]  # 基座
        
        current_T = np.eye(4, dtype=np.float64)
        for i, offset in enumerate(self.rest_translations):
            # 先平移至本关节原点
            T_trans = translation_matrix(offset)
            
            # 根据该关节的所有自由度构造旋转矩阵
            R_joint = np.eye(3, dtype=np.float64)
            axes = self.axes[i]
            angs = self.angles[i]
            for axis, ang in zip(axes, angs):
                if not np.isclose(ang, 0.0):
                    R_axis = rot_axis(axis, ang)
                    R_joint = R_axis @ R_joint  # 旋转叠加（注意顺序）
            
            T_rot = rotation_matrix_to_transform(R_joint)
            # 局部变换：平移 + 旋转
            T_local = T_trans @ T_rot
            
            # 累积到全局
            current_T = current_T @ T_local
            transforms.append(current_T.copy())
        
        return transforms

    def get_joint_positions(self, transforms: List[np.ndarray]) -> List[np.ndarray]:
        """
        从变换矩阵列表中提取各关节的世界坐标位置。
        
        参数:
            transforms: compute_transforms() 返回的变换矩阵列表
        返回:
            各关节的世界坐标列表（长度为关节数+1，基座原点为 [0,0,0]）
        """
        positions = []
        for T in transforms:
            positions.append(T[:3, 3].copy())
        return positions

    def get_end_effector_transform(self) -> np.ndarray:
        """
        获取末端执行器（指尖）的全局变换矩阵。
        """
        transforms = self.compute_transforms()
        return transforms[-1]


class HandKinematics:
    """
    整个手部的运动学模型，管理多根手指的运动链以及手掌的全局变换。
    """
    def __init__(self):
        # 手掌（基座）的全局变换
        self.palm_transform = np.eye(4, dtype=np.float64)
        # 存储各手指运动链
        self.fingers: Dict[str, KinematicChain] = {}

    def add_finger(self, name: str, chain: KinematicChain):
        """添加一根手指的运动链"""
        self.fingers[name] = chain

    def set_palm_transform(self, R: np.ndarray, t: np.ndarray):
        """设置手掌（基座）相对于世界坐标系的变换"""
        self.palm_transform = combine_transform(R, t)

    def compute_all_transforms(self) -> Dict[str, List[np.ndarray]]:
        """
        计算所有手指的全局变换矩阵（叠加手掌变换）。
        
        返回:
            字典，键为手指名称，值为该手指的全局变换矩阵列表。
        """
        all_transforms = {}
        for name, finger in self.fingers.items():
            # 先计算相对于手掌的局部变换
            local_transforms = finger.compute_transforms()
            # 应用手掌变换转换到世界坐标系
            world_transforms = [self.palm_transform @ T for T in local_transforms]
            all_transforms[name] = world_transforms
        return all_transforms

    def get_all_joint_positions(self) -> Dict[str, List[np.ndarray]]:
        """
        获取所有手指关节的世界坐标位置。
        
        返回:
            字典，键为手指名称，值为该手指各关节的世界坐标列表。
        """
        positions = {}
        all_transforms = self.compute_all_transforms()
        for name, transforms in all_transforms.items():
            pos_list = [T[:3, 3] for T in transforms]
            positions[name] = pos_list
        return positions
"""
模块二：手部几何定义 (hand_model.py)
功能：定义手部的骨骼长度、关节初始位置、旋转轴及自由度。
      构建符合人手的运动链模型，并暴露角度设置接口。
依赖：NumPy，以及模块一 kinematics
"""

import numpy as np
from kinematics import KinematicChain, HandKinematics, rot_x, rot_y, rot_z


class HandModel:
    """
    完整的灵巧手运动学模型。
    
    包括：
    - 手掌基座（位于世界原点）
    - 拇指（3个关节：CMC, MCP, IP）→ 3自由度
    - 食指、中指、无名指、小指（各3个关节：MCP, PIP, DIP）→ 每指3自由度
    
    总自由度 = 3 + 4*3 = 15 > 10，满足需求。
    """
    
    def __init__(self):
        # 初始化手部运动学容器
        self.hand_kin = HandKinematics()
        
        # 手部骨骼长度参数（单位：厘米，比例参考人手）
        # 手掌尺寸
        self.palm_width = 8.0       # 手掌宽度
        self.palm_length = 6.0      # 手掌长度（根部到中指MCP的距离）
        
        # 手指各指节长度（近节、中节、远节）
        self.finger_lengths = {
            'thumb': [3.0, 2.5, 2.0],      # 拇指：掌骨、近节、远节
            'index': [3.5, 2.8, 2.2],      # 食指：近节、中节、远节
            'middle': [3.8, 3.0, 2.4],     # 中指
            'ring': [3.5, 2.8, 2.2],       # 无名指
            'pinky': [2.8, 2.2, 1.8]       # 小指
        }
        
        # 手指根部在手掌上的偏移位置（相对于手掌坐标系原点）
        # 手掌坐标系定义：原点位于手腕中心，X轴向右，Y轴向前（手指方向），Z轴向上
        self.finger_base_offsets = {
            'thumb':   np.array([-2.5, -1.0, 0.0]),   # 拇指在手掌外侧偏下
            'index':   np.array([-1.5,  0.0, 0.0]),   # 食指
            'middle':  np.array([ 0.0,  0.5, 0.0]),   # 中指（稍前）
            'ring':    np.array([ 1.5,  0.0, 0.0]),   # 无名指
            'pinky':   np.array([ 2.8, -0.8, 0.0])    # 小指
        }
        
        # 构建所有手指的运动链
        self._build_thumb()
        self._build_finger('index')
        self._build_finger('middle')
        self._build_finger('ring')
        self._build_finger('pinky')
        
        # 存储所有关节名称（用于界面交互）
        self.joint_names = self._collect_joint_names()
    
    def _build_thumb(self):
        """
        构建拇指运动链。
        
        拇指关节：
        - CMC (腕掌关节)：2自由度，屈曲/伸展 (绕X轴) + 外展/内收 (绕Z轴)
        - MCP (掌指关节)：1自由度，屈曲/伸展 (绕X轴)
        - IP  (指间关节)：1自由度，屈曲/伸展 (绕X轴)
        
        注意：为了模拟拇指的对掌动作，CMC关节的旋转轴需要精心设置。
        这里将CMC置于手掌坐标系下的特定位置，使其旋转后能与其他手指相对。
        """
        chain = KinematicChain('thumb')
        
        # 定义各关节的默认旋转轴（单位向量）
        # CMC: 先外展(Z)后屈曲(X)的顺序由 KinematicChain 中的连乘顺序决定（后添加的先作用？）
        # 这里 axes 列表的顺序就是旋转矩阵相乘的顺序：R = ... @ R_axis2 @ R_axis1
        # 我们希望先外展后屈曲，所以列表第一个是外展轴(Z)，第二个是屈曲轴(X)
        axes_cmc = [np.array([0, 0, 1]), np.array([1, 0, 0])]   # Z轴外展，X轴屈曲
        axes_mcp = [np.array([1, 0, 0])]                        # X轴屈曲
        axes_ip  = [np.array([1, 0, 0])]                        # X轴屈曲
        
        # 获取拇指各节长度
        l0, l1, l2 = self.finger_lengths['thumb']
        
        # 添加关节：
        # CMC 关节：从手掌拇指根部偏移量开始，旋转中心位于手掌附着点
        offset_cmc = self.finger_base_offsets['thumb']
        chain.add_joint(offset_cmc, axes_cmc)
        
        # MCP 关节：沿着拇指掌骨方向平移 l0
        # 注意：拇指掌骨的初始方向是沿Y轴偏外侧，这里简化设为沿Y轴方向，后续由CMC旋转调整
        offset_mcp = np.array([0, l0, 0])
        chain.add_joint(offset_mcp, axes_mcp)
        
        # IP 关节：沿着近节指骨方向平移 l1
        offset_ip = np.array([0, l1, 0])
        chain.add_joint(offset_ip, axes_ip)
        
        # 指尖位置：添加一个虚拟关节来表示指尖点，无自由度，仅有平移 l2
        offset_tip = np.array([0, l2, 0])
        chain.add_joint(offset_tip, [])   # 无旋转自由度
        
        self.hand_kin.add_finger('thumb', chain)
    
    def _build_finger(self, name: str):
        """
        构建普通四指（食指、中指、无名指、小指）的运动链。
        
        每个手指包含三个关节：
        - MCP (掌指关节)：2自由度，屈曲/伸展 (绕X轴) + 外展/内收 (绕Z轴)
        - PIP (近端指间关节)：1自由度，屈曲/伸展 (绕X轴)
        - DIP (远端指间关节)：1自由度，屈曲/伸展 (绕X轴)
        """
        chain = KinematicChain(name)
        
        # 关节轴定义
        axes_mcp = [np.array([1, 0, 0]), np.array([0, 0, 1])]   # X屈曲，Z外展
        axes_pip = [np.array([1, 0, 0])]
        axes_dip = [np.array([1, 0, 0])]
        
        # 获取指节长度
        l0, l1, l2 = self.finger_lengths[name]
        base_offset = self.finger_base_offsets[name]
        
        # 添加关节
        chain.add_joint(base_offset, axes_mcp)                 # MCP
        chain.add_joint(np.array([0, l0, 0]), axes_pip)        # PIP
        chain.add_joint(np.array([0, l1, 0]), axes_dip)        # DIP
        chain.add_joint(np.array([0, l2, 0]), [])              # 指尖
        
        self.hand_kin.add_finger(name, chain)
    
    def _collect_joint_names(self) -> list:
        """
        收集所有可调节的关节名称，格式为 "手指_关节索引"。
        关节索引：0 表示第一个关节（CMC或MCP），1表示第二个，依此类推。
        """
        names = []
        for finger_name, chain in self.hand_kin.fingers.items():
            for i in range(len(chain.angles)):
                # 跳过无自由度的虚拟指尖关节
                if len(chain.axes[i]) > 0:
                    names.append(f"{finger_name}_{i}")
        return names
    
    # -------------------- 角度设置接口 --------------------
    def set_joint_angle(self, finger: str, joint_idx: int, dof_idx: int, angle_rad: float):
        """
        设置某个关节的特定自由度的角度。
        
        参数:
            finger: 手指名称 ('thumb', 'index', ...)
            joint_idx: 关节索引 (0: CMC/MCP, 1: MCP/PIP, 2: IP/DIP)
            dof_idx: 该关节内的自由度索引 (0 或 1)
            angle_rad: 角度值（弧度）
        """
        chain = self.hand_kin.fingers.get(finger)
        if chain is None:
            raise ValueError(f"手指 '{finger}' 不存在")
        if joint_idx >= len(chain.angles):
            raise ValueError(f"关节索引 {joint_idx} 超出范围")
        
        angles = chain.angles[joint_idx].copy()
        if dof_idx >= len(angles):
            raise ValueError(f"自由度索引 {dof_idx} 超出范围")
        
        angles[dof_idx] = angle_rad
        chain.set_joint_angles(joint_idx, angles)
    
    def set_finger_angles(self, finger: str, joint_angles: list):
        """
        设置某根手指所有关节的角度（列表格式）。
        
        参数:
            finger: 手指名称
            joint_angles: 列表，每个元素为该关节的自由度角度列表
                          例如 [[mcp_flex, mcp_abd], [pip_flex], [dip_flex]]
        """
        chain = self.hand_kin.fingers.get(finger)
        if chain is None:
            raise ValueError(f"手指 '{finger}' 不存在")
        for i, ang in enumerate(joint_angles):
            if i < len(chain.angles):
                chain.set_joint_angles(i, ang)
    
    def reset_pose(self):
        """
        将所有关节角度重置为0（伸直状态）。
        """
        for chain in self.hand_kin.fingers.values():
            for i in range(len(chain.angles)):
                chain.set_joint_angles(i, [0.0] * len(chain.angles[i]))
    
    # -------------------- 获取变换和位置 --------------------
    def get_all_transforms(self):
        """获取所有手指的全局变换矩阵（调用 HandKinematics 方法）"""
        return self.hand_kin.compute_all_transforms()
    
    def get_all_joint_positions(self):
        """获取所有关节的世界坐标位置"""
        return self.hand_kin.get_all_joint_positions()
    
    def get_finger_joint_positions(self, finger: str):
        """获取单根手指的关节坐标列表"""
        all_pos = self.hand_kin.get_all_joint_positions()
        return all_pos.get(finger, [])
    
    def get_joint_names(self):
        """返回所有可调节关节的名称列表"""
        return self.joint_names
    
    # -------------------- 预设手势 --------------------
    def apply_gesture(self, gesture_name: str):
        """
        应用预设手势（为演示方便）。
        """
        self.reset_pose()
        if gesture_name == 'thumbs_up':
            # 竖大拇指：拇指伸直，其余四指握拳
            # 拇指伸直 (角度0即可)
            # 其余四指弯曲 MCP 和 PIP
            for finger in ['index', 'middle', 'ring', 'pinky']:
                self.set_joint_angle(finger, 0, 0, np.radians(80))   # MCP 屈曲
                self.set_joint_angle(finger, 1, 0, np.radians(90))   # PIP 屈曲
                self.set_joint_angle(finger, 2, 0, np.radians(60))   # DIP 屈曲
            # 拇指稍外展使其更明显
            self.set_joint_angle('thumb', 0, 1, np.radians(30))      # CMC 外展
            self.set_joint_angle('thumb', 0, 0, np.radians(-10))     # CMC 伸展
            
        elif gesture_name == 'fist':
            # 握拳：所有手指弯曲
            for finger in self.hand_kin.fingers.keys():
                if finger == 'thumb':
                    self.set_joint_angle('thumb', 0, 0, np.radians(50))
                    self.set_joint_angle('thumb', 0, 1, np.radians(20))
                    self.set_joint_angle('thumb', 1, 0, np.radians(40))
                    self.set_joint_angle('thumb', 2, 0, np.radians(30))
                else:
                    self.set_joint_angle(finger, 0, 0, np.radians(85))
                    self.set_joint_angle(finger, 1, 0, np.radians(100))
                    self.set_joint_angle(finger, 2, 0, np.radians(70))
                    
        elif gesture_name == 'ok':
            # OK手势：拇指与食指指尖相触，其余手指伸直
            # 这里简化：食指微曲，拇指旋转对指，需要手动调整参数使指尖接近
            self.set_joint_angle('thumb', 0, 1, np.radians(40))   # 外展
            self.set_joint_angle('thumb', 0, 0, np.radians(30))   # 屈曲
            self.set_joint_angle('thumb', 1, 0, np.radians(20))
            self.set_joint_angle('thumb', 2, 0, np.radians(20))
            
            self.set_joint_angle('index', 0, 0, np.radians(60))
            self.set_joint_angle('index', 1, 0, np.radians(40))
            self.set_joint_angle('index', 2, 0, np.radians(20))
            
            for finger in ['middle', 'ring', 'pinky']:
                self.set_joint_angle(finger, 0, 0, np.radians(10))  # 轻微弯曲
            
        elif gesture_name == 'point':
            # 食指伸出，其余握拳
            for finger in ['middle', 'ring', 'pinky']:
                self.set_joint_angle(finger, 0, 0, np.radians(85))
                self.set_joint_angle(finger, 1, 0, np.radians(90))
                self.set_joint_angle(finger, 2, 0, np.radians(60))
            self.set_joint_angle('thumb', 0, 0, np.radians(50))
            self.set_joint_angle('thumb', 0, 1, np.radians(20))
            self.set_joint_angle('thumb', 1, 0, np.radians(40))
            self.set_joint_angle('thumb', 2, 0, np.radians(30))
            # 食指伸直 (角度0)
        
        else:
            print(f"未知手势: {gesture_name}")


# -------------------- 测试代码（可选）--------------------
if __name__ == "__main__":
    # 简单测试：创建手部模型并输出各关节位置
    hand = HandModel()
    hand.reset_pose()
    positions = hand.get_all_joint_positions()
    
    print("各手指关节世界坐标（初始伸直状态）:")
    for finger, pos_list in positions.items():
        print(f"  {finger}:")
        for i, p in enumerate(pos_list):
            print(f"    关节{i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    
    # 测试竖大拇指手势
    print("\n应用 'thumbs_up' 手势...")
    hand.apply_gesture('thumbs_up')
    positions = hand.get_all_joint_positions()
    print("拇指关节坐标：")
    for i, p in enumerate(positions['thumb']):
        print(f"  关节{i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
"""
模块三：交互控制与姿态解算 (controller.py)
功能：提供用户交互接口，包括：
      - 滑块调节各关节角度（直接控制）
      - 按钮切换预设手势
      - 逆向运动学求解器框架（预留）
依赖：NumPy，模块一 kinematics，模块二 hand_model
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from hand_model import HandModel
from kinematics import rot_axis, transform_inverse


class HandController:
    """
    手部姿态控制器，负责管理角度状态并连接模型与UI。
    
    该控制器设计为与可视化模块协同工作：
    - 维护当前各关节角度值
    - 响应滑块变化事件，更新模型并触发重绘
    - 提供手势预设接口
    """
    
    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        # 缓存所有可调节关节的信息：[(手指名, 关节索引, 自由度索引, 当前角度, 轴向量), ...]
        self.joint_params = self._extract_joint_params()
        
        # 滑块对象列表（由外部创建后填充）
        self.sliders: List[Slider] = []
        # 回调函数：当模型更新后调用，用于刷新可视化
        self.update_callback: Optional[Callable[[], None]] = None
    
    def _extract_joint_params(self) -> List[Tuple[str, int, int, float, np.ndarray]]:
        """
        从模型中提取所有可调节关节的信息。
        
        返回列表，每项为 (手指名, 关节索引, 自由度索引, 初始角度, 旋转轴)
        """
        params = []
        for finger_name, chain in self.model.hand_kin.fingers.items():
            for joint_idx in range(len(chain.angles)):
                axes = chain.axes[joint_idx]
                angles = chain.angles[joint_idx]
                for dof_idx, (axis, angle) in enumerate(zip(axes, angles)):
                    params.append((finger_name, joint_idx, dof_idx, angle, axis.copy()))
        return params
    
    def set_update_callback(self, callback: Callable[[], None]):
        """设置模型更新后的回调函数（通常用于重绘画面）"""
        self.update_callback = callback
    
    def get_joint_angle(self, finger: str, joint_idx: int, dof_idx: int) -> float:
        """获取指定关节自由度的当前角度（弧度）"""
        chain = self.model.hand_kin.fingers[finger]
        return chain.angles[joint_idx][dof_idx]
    
    def set_joint_angle(self, finger: str, joint_idx: int, dof_idx: int, angle_rad: float):
        """
        设置关节角度并更新模型。
        如果设置了回调，会自动触发重绘。
        """
        self.model.set_joint_angle(finger, joint_idx, dof_idx, angle_rad)
        # 同步更新内部缓存的角度值
        for i, (f, j, d, _, _) in enumerate(self.joint_params):
            if f == finger and j == joint_idx and d == dof_idx:
                self.joint_params[i] = (f, j, d, angle_rad, self.joint_params[i][4])
                break
        if self.update_callback:
            self.update_callback()
    
    def apply_gesture(self, gesture_name: str):
        """
        应用预设手势，并同步更新所有滑块的值。
        """
        self.model.apply_gesture(gesture_name)
        # 更新内部缓存的角度值
        for i, (f, j, d, _, axis) in enumerate(self.joint_params):
            new_angle = self.model.hand_kin.fingers[f].angles[j][d]
            self.joint_params[i] = (f, j, d, new_angle, axis)
        # 更新滑块位置（如果滑块已创建）
        self._sync_sliders()
        if self.update_callback:
            self.update_callback()
    
    def _sync_sliders(self):
        """将内部角度缓存同步到滑块控件"""
        for slider, (_, _, _, angle, _) in zip(self.sliders, self.joint_params):
            slider.set_val(np.degrees(angle))  # 滑块通常显示角度值
    
    def create_slider_callback(self, finger: str, joint_idx: int, dof_idx: int):
        """
        生成一个滑块回调函数，用于响应滑块值变化。
        返回的函数符合 matplotlib Slider 的 on_changed 签名。
        """
        def callback(val_deg: float):
            angle_rad = np.radians(val_deg)
            self.set_joint_angle(finger, joint_idx, dof_idx, angle_rad)
        return callback
    
    def get_slider_label(self, finger: str, joint_idx: int, dof_idx: int) -> str:
        """生成滑块标签，例如 'index_MCP_flex'"""
        joint_names = {0: 'CMC', 1: 'MCP', 2: 'IP'} if finger == 'thumb' else {0: 'MCP', 1: 'PIP', 2: 'DIP'}
        dof_names = {0: 'flex', 1: 'abd'}
        joint_label = joint_names.get(joint_idx, f'J{joint_idx}')
        dof_label = dof_names.get(dof_idx, f'dof{dof_idx}')
        return f"{finger}_{joint_label}_{dof_label}"
    
    def get_joint_count(self) -> int:
        """返回可调节自由度的总数"""
        return len(self.joint_params)


class IKController:
    """
    逆向运动学求解器框架（用于高级交互，如指尖触碰、抓取）。
    
    基于雅可比矩阵伪逆的迭代求解方法。
    该类可独立于 HandController 使用，也可集成到交互中。
    """
    
    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        self.damping = 0.01          # 阻尼系数，防止奇异
        self.max_iter = 50           # 最大迭代次数
        self.tolerance = 1e-3        # 位置误差容限
    
    def compute_jacobian(self, 
                         finger: str, 
                         target_link_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算指定手指末端（或指定连杆）相对于各可调自由度的雅可比矩阵。
        
        采用数值差分法（简单可靠）：
        J[:, i] = (f(θ + Δθ_i) - f(θ)) / Δθ
        
        参数:
            finger: 手指名称
            target_link_idx: 目标连杆索引，-1 表示指尖
        返回:
            J: 3 x n 雅可比矩阵，n 为该手指的自由度数量
            current_pos: 当前目标点的世界坐标 (3,)
        """
        chain = self.model.hand_kin.fingers[finger]
        # 获取该手指的所有自由度参数索引（相对于全局 joint_params 的切片）
        # 这里简化：只考虑该手指自身关节的自由度
        dof_count = sum(len(axes) for axes in chain.axes)
        if dof_count == 0:
            return np.zeros((3, 0)), np.zeros(3)
        
        # 保存当前角度
        original_angles = [angles.copy() for angles in chain.angles]
        
        # 计算当前位置
        transforms = self.model.hand_kin.compute_all_transforms()[finger]
        current_pos = transforms[target_link_idx][:3, 3].copy()
        
        # 差分步长
        delta = 1e-4
        
        J = np.zeros((3, dof_count))
        col = 0
        for joint_idx, angles in enumerate(original_angles):
            for dof_idx in range(len(angles)):
                # 正向扰动
                chain.angles[joint_idx][dof_idx] += delta
                transforms_plus = self.model.hand_kin.compute_all_transforms()[finger]
                pos_plus = transforms_plus[target_link_idx][:3, 3]
                # 恢复
                chain.angles[joint_idx][dof_idx] = original_angles[joint_idx][dof_idx]
                # 计算偏导数
                J[:, col] = (pos_plus - current_pos) / delta
                col += 1
        
        # 恢复所有角度（确保状态不变）
        for joint_idx, angles in enumerate(original_angles):
            chain.angles[joint_idx] = angles.copy()
        
        return J, current_pos
    
    def solve_ik(self, 
                 finger: str, 
                 target_pos: np.ndarray, 
                 target_link_idx: int = -1) -> bool:
        """
        使用雅可比伪逆迭代求解逆向运动学，使指定指尖到达目标位置。
        
        参数:
            finger: 手指名称
            target_pos: 目标世界坐标 (3,)
            target_link_idx: 目标连杆索引，-1 表示指尖
        返回:
            bool: 是否成功收敛
        """
        chain = self.model.hand_kin.fingers[finger]
        
        for iteration in range(self.max_iter):
            J, current_pos = self.compute_jacobian(finger, target_link_idx)
            error = target_pos - current_pos
            if np.linalg.norm(error) < self.tolerance:
                return True
            
            if J.shape[1] == 0:
                break
            
            # 阻尼最小二乘 (伪逆 + 阻尼)
            JJT = J @ J.T
            lambda_sq = self.damping ** 2
            # 求解 Δθ = J^T (J J^T + λ^2 I)^{-1} error
            try:
                delta_theta = J.T @ np.linalg.solve(JJT + lambda_sq * np.eye(3), error)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                delta_theta = np.linalg.pinv(J) @ error
            
            # 更新关节角度
            col = 0
            for joint_idx, angles in enumerate(chain.angles):
                for dof_idx in range(len(angles)):
                    chain.angles[joint_idx][dof_idx] += delta_theta[col]
                    col += 1
        
        return False


# -------------------- UI 构建辅助函数 --------------------
def setup_interactive_controls(hand_model: HandModel, 
                               fig: plt.Figure, 
                               update_view_callback: Callable[[], None]) -> HandController:
    """
    在给定的 matplotlib 图像上创建交互控件（滑块和按钮），并返回控制器。
    
    参数:
        hand_model: 手部模型实例
        fig: matplotlib Figure 对象
        update_view_callback: 更新可视化画面的回调函数（无参数）
    返回:
        HandController 实例，已绑定滑块和回调
    """
    controller = HandController(hand_model)
    controller.set_update_callback(update_view_callback)
    
    # 调整主图区域，为滑块留出空间
    plt.subplots_adjust(bottom=0.35)
    
    # 创建滑块
    slider_axes = []
    num_sliders = controller.get_joint_count()
    # 计算滑块布局：每行最多4个滑块
    n_cols = 4
    n_rows = (num_sliders + n_cols - 1) // n_cols
    slider_height = 0.03
    slider_spacing = 0.01
    bottom_start = 0.25
    slider_width = 0.15
    left_margin = 0.1
    
    for i, (finger, joint_idx, dof_idx, angle_rad, _) in enumerate(controller.joint_params):
        row = i // n_cols
        col = i % n_cols
        ax_slider = fig.add_axes([
            left_margin + col * (slider_width + 0.05),
            bottom_start - row * (slider_height + slider_spacing),
            slider_width,
            slider_height
        ])
        label = controller.get_slider_label(finger, joint_idx, dof_idx)
        # 滑块范围：-90° 到 90°（可调）
        slider = Slider(ax_slider, label, -90.0, 90.0, 
                        valinit=np.degrees(angle_rad), valstep=1.0)
        slider.on_changed(controller.create_slider_callback(finger, joint_idx, dof_idx))
        controller.sliders.append(slider)
        slider_axes.append(ax_slider)
    
    # 创建手势按钮
    button_axes = []
    gestures = ['thumbs_up', 'fist', 'ok', 'point', 'reset']
    button_width = 0.08
    button_height = 0.04
    button_start_y = 0.15
    for i, gesture in enumerate(gestures):
        ax_btn = fig.add_axes([
            left_margin + i * (button_width + 0.02),
            button_start_y,
            button_width,
            button_height
        ])
        btn = Button(ax_btn, gesture.replace('_', ' ').title())
        # 定义按钮回调
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
        button_axes.append(ax_btn)
    
    return controller


# -------------------- 测试代码 --------------------
if __name__ == "__main__":
    # 简单测试：创建模型和控制器，手动调节角度
    from hand_model import HandModel
    import matplotlib.pyplot as plt
    
    hand = HandModel()
    controller = HandController(hand)
    
    print(f"共有 {controller.get_joint_count()} 个可调节自由度")
    for i, (f, j, d, a, _) in enumerate(controller.joint_params):
        print(f"{i}: {f} joint {j} dof {d} = {np.degrees(a):.1f}°")
    
    # 测试角度设置
    controller.set_joint_angle('index', 0, 0, np.radians(45))
    print("\n设置食指 MCP 屈曲 45° 后：")
    pos = hand.get_finger_joint_positions('index')
    for i, p in enumerate(pos):
        print(f"  关节{i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
"""
模块四：可视化渲染管线 (visualizer.py)
功能：使用 matplotlib 3D 将手部模型渲染为点线结构，并响应控制器更新。
     提供交互式窗口，支持鼠标拖拽旋转视角。
依赖：NumPy, Matplotlib, 模块二 hand_model, 模块三 controller
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from typing import Dict, List, Tuple, Optional

from hand_model import HandModel
from controller import HandController, setup_interactive_controls


class HandVisualizer:
    """
    手部3D可视化器。
    
    负责：
    - 创建 3D 坐标轴
    - 根据手部关节位置绘制点（关节）和线（骨骼）
    - 在控制器更新时刷新画面
    """
    
    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        self.fig = None
        self.ax: Optional[Axes3D] = None
        
        # 存储绘制对象的字典
        self.joint_scatter = None          # 关节点散点图
        self.bone_lines: Dict[str, List[Line3D]] = {}  # 骨骼线条，按手指分组
        self.palm_lines: List[Line3D] = []             # 手掌轮廓线
        
        # 视觉参数
        self.joint_color = 'red'
        self.joint_size = 30
        self.bone_color_map = {
            'thumb': 'blue',
            'index': 'green',
            'middle': 'orange',
            'ring': 'purple',
            'pinky': 'brown'
        }
        self.palm_color = 'gray'
        self.line_width = 2.0
        
    def setup_figure(self, figsize=(10, 8)):
        """创建 matplotlib 图像和 3D 坐标轴"""
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title('机械灵巧手 - 点线模型')
        
        # 设置初始视角
        self.ax.view_init(elev=20, azim=-60)
        
        # 设置坐标轴范围（根据手部大致尺寸预估）
        self.ax.set_xlim([-5, 10])
        self.ax.set_ylim([-2, 18])
        self.ax.set_zlim([-5, 10])
        
        # 使坐标轴等比例（尽量）
        self.ax.set_box_aspect([1, 1, 1])
        
    def draw_hand(self):
        """
        根据当前模型状态绘制或更新手部图形。
        首次调用时创建图形对象，后续调用更新数据。
        """
        if self.ax is None:
            raise RuntimeError("请先调用 setup_figure() 创建图形窗口")
        
        # 获取所有关节世界坐标
        positions = self.model.get_all_joint_positions()
        
        # 收集所有关节点坐标（用于散点图）
        all_points = []
        
        # 1. 更新或创建骨骼线条
        for finger_name, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            
            # 提取关节点坐标数组 (N, 3)
            points = np.array(pos_list)
            all_points.append(points)
            
            # 生成线段：相邻关节点连接
            segments = [(points[i], points[i+1]) for i in range(len(points)-1)]
            
            # 获取该手指对应的颜色
            color = self.bone_color_map.get(finger_name, 'black')
            
            if finger_name not in self.bone_lines:
                # 首次创建：为每段骨骼创建一个 Line3D 对象
                lines = []
                for seg in segments:
                    line = self.ax.plot([seg[0][0], seg[1][0]],
                                        [seg[0][1], seg[1][1]],
                                        [seg[0][2], seg[1][2]],
                                        color=color, linewidth=self.line_width)[0]
                    lines.append(line)
                self.bone_lines[finger_name] = lines
            else:
                # 更新已有线条的数据
                lines = self.bone_lines[finger_name]
                for i, seg in enumerate(segments):
                    if i < len(lines):
                        lines[i].set_data([seg[0][0], seg[1][0]],
                                          [seg[0][1], seg[1][1]])
                        lines[i].set_3d_properties([seg[0][2], seg[1][2]])
        
        # 2. 绘制手掌轮廓（连接各手指根部，形成手掌多边形）
        #    获取各手指的第一个关节位置（即 MCP 或 CMC 根部）
        root_positions = {}
        for finger_name, pos_list in positions.items():
            if len(pos_list) > 0:
                root_positions[finger_name] = pos_list[0]
        
        # 按顺序连接：拇指 -> 食指 -> 中指 -> 无名指 -> 小指
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        palm_points = []
        for f in finger_order:
            if f in root_positions:
                palm_points.append(root_positions[f])
        
        if len(palm_points) >= 2:
            palm_segments = [(palm_points[i], palm_points[i+1]) for i in range(len(palm_points)-1)]
            # 添加闭合线段（小指到拇指）
            if len(palm_points) >= 4:
                palm_segments.append((palm_points[-1], palm_points[0]))
            
            if not self.palm_lines:
                for seg in palm_segments:
                    line = self.ax.plot([seg[0][0], seg[1][0]],
                                        [seg[0][1], seg[1][1]],
                                        [seg[0][2], seg[1][2]],
                                        color=self.palm_color, linewidth=self.line_width,
                                        linestyle='--')[0]
                    self.palm_lines.append(line)
            else:
                for i, seg in enumerate(palm_segments):
                    if i < len(self.palm_lines):
                        self.palm_lines[i].set_data([seg[0][0], seg[1][0]],
                                                    [seg[0][1], seg[1][1]])
                        self.palm_lines[i].set_3d_properties([seg[0][2], seg[1][2]])
        
        # 3. 更新关节点散点图
        all_points = np.vstack(all_points) if all_points else np.empty((0, 3))
        
        if self.joint_scatter is None:
            self.joint_scatter = self.ax.scatter(all_points[:, 0],
                                                 all_points[:, 1],
                                                 all_points[:, 2],
                                                 c=self.joint_color,
                                                 s=self.joint_size,
                                                 depthshade=True)
        else:
            # 更新散点位置
            self.joint_scatter._offsets3d = (all_points[:, 0],
                                             all_points[:, 1],
                                             all_points[:, 2])
        
        # 刷新画布
        self.fig.canvas.draw_idle()
    
    def update_view(self):
        """
        由控制器调用的更新函数。
        重新计算手部位置并刷新图形。
        """
        self.draw_hand()
    
    def show(self):
        """显示图形窗口（阻塞）"""
        if self.fig is not None:
            plt.show()
    
    def animate(self, controller: HandController):
        """
        启动交互模式：连接控制器并显示窗口。
        此函数会调用 plt.show()，进入 matplotlib 事件循环。
        
        参数:
            controller: 已配置好回调的 HandController 实例
        """
        # 确保控制器回调指向本可视化器的更新方法
        controller.set_update_callback(self.update_view)
        
        # 初始绘制
        self.draw_hand()
        
        # 显示窗口
        self.show()


def create_interactive_hand():
    """
    一键创建完整的交互式灵巧手仿真。
    
    返回:
        fig, model, controller, visualizer
    """
    # 1. 创建手部模型
    hand = HandModel()
    
    # 2. 创建可视化器并设置图形
    viz = HandVisualizer(hand)
    viz.setup_figure(figsize=(12, 8))
    
    # 3. 创建控制器并绑定滑块（注意：控制器内部会调整 figure 布局）
    controller = setup_interactive_controls(hand, viz.fig, viz.update_view)
    
    # 4. 初始绘制
    viz.draw_hand()
    
    return viz.fig, hand, controller, viz


# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    print("正在启动机械灵巧手仿真...")
    print("使用滑块调节关节角度，按钮切换预设手势。")
    print("鼠标拖拽可旋转视角。")
    
    # 创建交互式手部仿真
    fig, hand, controller, viz = create_interactive_hand()
    
    # 显示窗口（阻塞，直到关闭）
    plt.show()
    
    print("仿真结束。")