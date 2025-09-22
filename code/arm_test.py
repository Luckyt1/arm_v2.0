#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
右手数据到机械臂映射的ROS2节点
基于vmc.py读取右手位置和姿态数据，映射到7自由度机械臂，并通过ROS2发布qpos数据
"""

import numpy as np
import signal
import sys
import time
import threading
import struct
import select
import os
import fcntl
from math import radians, degrees
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import communication.msg as bxiMsg
import usr_communication.msg as usrMsg




# VMC相关导入
from vmcp.osc import OSC
from vmcp.osc.typing import Message
from vmcp.osc.backend.osc4py3 import as_comthreads as backend
from vmcp.events import (
    Event,
    RootTransformEvent,
    BoneTransformEvent,
    BlendShapeEvent,
    BlendShapeApplyEvent,
    DeviceTransformEvent,
    StateEvent,
    RelativeTimeEvent
)
from vmcp.typing import (
    CoordinateVector,
    Quaternion,
    Bone,
    DeviceType,
    BlendShapeKey as AbstractBlendShapeKey,
    ModelState,
    Timestamp
)
from vmcp.protocol import (
    root_transform,
    bone_transform,
    device_transform,
    blendshape,
    blendshape_apply,
    state,
)
from vmcp.facades import on_receive

# from pynput import keyboard

# 手柄配置常量 - 可根据实际情况修改
GAMEPAD_DEVICES = [
    "/dev/input/vr_js0",  # 第一个手柄，通常是左手
    "/dev/input/vr_js1",  # 第二个手柄，通常是右手
]

# 手柄按键和轴配置
BUTTON_0 = 0        # 按键0，用于夹爪控制
BUTTON_4 = 4        # 按键4，用于夹爪控制
AXIS_4 = 4          # 轴4，摇杆上下（上负下正）
AXIS_5 = 5          # 轴5，摇杆左右（左正右负）

# 摇杆阈值（摇杆触发时的值）
STICK_THRESHOLD = 30000  # 摇杆最大值约为32767
GRIPPER_STEP = 0.5       # 夹爪每次移动的步长

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_z_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_y_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_z_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_y_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shld_y_joint",   # 左臂_肩关节_y轴
    "l_shld_x_joint",   # 左臂_肩关节_x轴
    "l_shld_z_joint",   # 左臂_肩关节_z轴
    "l_elb_y_joint",   # 左臂_肘关节_y轴
    "l_elb_z_joint",   # 左臂_肘关节_y轴
    
    "r_shld_y_joint",   # 右臂_肩关节_y轴   
    "r_shld_x_joint",   # 右臂_肩关节_x轴
    "r_shld_z_joint",   # 右臂_肩关节_z轴
    "r_elb_y_joint",    # 右臂_肘关节_y轴
    "r_elb_z_joint",    # 右臂_肘关节_y轴
    
    "l_wrist_y_joint",
    "l_wrist_x_joint",
    "l_hand_joint",

    "r_wrist_y_joint",
    "r_wrist_x_joint",
    "r_hand_joint",    
    )   

joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0.1,0.0,0.0,-0.3,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.1,0.0,0.0,-0.3,0.0,
    0,0,0,
    0,0,0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    500,500,300,
    100,100,100,150,30,10,
    100,100,100,150,30,10,
    40,50,15,40,15,
    40,50,15,40,15,
    15,15,10,
    15,15,10,], dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    5,5,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    1.0,1.0,0.8,1.0,0.8,
    1.0,1.0,0.8,1.0,0.8,
    0.4,0.4,0.5,
    0.4,0.4,0.5], dtype=np.float32)

#循环限幅
def limit_angle_range(angle, min_angle=-np.pi, max_angle=np.pi):
    """
    将角度限制在指定范围内，支持循环限幅
    :param angle: 输入角度
    :param min_angle: 最小角度
    :param max_angle: 最大角度
    :return: 限制后的角度
    """
    range_size = max_angle - min_angle
    while angle < min_angle:
        angle += range_size
    while angle > max_angle:
        angle -= range_size
    return angle

def quaternion_to_euler(quat, degrees_output=False, seq='xyz'):
    """
    将四元数转换为欧拉角
    :param quat: 四元数，格式为[x, y, z, w]或[w, x, y, z]
    :param degrees_output: 是否输出角度制
    :param seq: 欧拉角顺序，默认'xyz'
    :return: 欧拉角 (roll, pitch, yaw)
    """
    # 确保输入是numpy数组
    quat = np.array(quat)
    # 如果是[w, x, y, z]格式，转换为[x, y, z, w]
    if quat.shape[-1] == 4 and abs(quat[0]) > 1 and abs(quat[3]) <= 1:
        quat = np.roll(quat, -1)
    r = Rotation.from_quat(quat)
    euler = r.as_euler(seq, degrees=degrees_output)
    
    return euler

def unwrap_angle(angle, last_angle):
    """
    将一个角度解缠绕，使其与上一个角度的差值在[-pi, pi]之间。
    这能选择与上一帧最接近的 2k*pi 等价解。
    """
    return last_angle + (angle - last_angle + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_xyx_euler_continuous(q, last_euler=np.array([0.0, 0.0, 0.0])):
    """
    将四元数平滑地转换为XYX欧拉角，通过选择与上一帧解最接近的等价解来避免跳变。
    
    这个版本综合处理奇点和2*pi周期性问题。

    参数:
    q (tuple or np.ndarray): 当前的四元数 (x, y, z, w)。
    last_euler (np.ndarray): 上一帧的欧拉角 [alpha, beta, gamma]，用于解缠绕。

    返回:
    np.ndarray: 计算出的最平滑的欧拉角 [alpha, beta, gamma]。
    """
    x, y, z, w = q
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-9: return last_euler # 如果四元数无效，则返回上一帧的值
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # --- 步骤 1: 计算两组可能的解 ---

    # 从旋转矩阵元素计算
    # R[0,0] = cos(beta)
    r00 = 1 - 2 * (y**2 + z**2)
    # R[0,1] = sin(beta)sin(gamma)
    r01 = 2 * (x*y - w*z)
    # R[0,2] = sin(beta)cos(gamma)
    r02 = 2 * (x*z + w*y)
    # R[1,0] = sin(beta)sin(alpha)
    r10 = 2 * (x*y + w*z)
    # R[2,0] = -sin(beta)cos(alpha)
    r20 = 2 * (x*z - w*y)

    # 第一组解 (标准解，beta in [0, pi])
    beta1 = np.arccos(r00)
    sin_beta = np.sin(beta1)
    
    # 设定一个小的阈值来判断是否在奇点
    epsilon = 1e-7
    
    if abs(sin_beta) > epsilon:
        # 非奇点情况
        alpha1 = np.arctan2(r10, -r20)
        gamma1 = np.arctan2(r01, r02)
    else:
        # 奇点情况: β ≈ 0 或 β ≈ π
        # 此时 α 和 γ 耦合，我们只能确定它们的和或差
        # 约定俗成地，将 γ 设为上一帧的值，以保持连续性
        # (α ± γ) = atan2(R[2,1], R[1,1])
        r21 = 2 * (y*z + w*x)
        r11 = 1 - 2 * (x**2 + z**2)
        total_angle = np.arctan2(r21, r11)

        gamma1 = last_euler[2] # 锁定 gamma
        if beta1 < np.pi / 2: # beta ≈ 0
            alpha1 = total_angle - gamma1
        else: # beta ≈ pi
            alpha1 = total_angle + gamma1
            
    solution1 = np.array([alpha1, beta1, gamma1])

    # 第二组解 (数学等价解)
    # α' = α + π, β' = -β, γ' = γ + π
    # 但由于我们希望 β 在 [0, pi] 范围内，所以我们用 2pi - beta'
    # 并且需要对 alpha 和 gamma 做相应调整以保持等价
    # 更简单的方法是直接从矩阵中找到beta相反的解
    beta2 = -beta1
    alpha2 = np.arctan2(-r10, r20)
    gamma2 = np.arctan2(-r01, -r02)
    solution2 = np.array([alpha2, beta2, gamma2])

    # --- 步骤 2: 解缠绕并选择最佳解 ---
    
    # 对两组解的每个角度进行解缠绕，使其最接近上一帧
    sol1_unwrapped = np.array([
        unwrap_angle(solution1[0], last_euler[0]),
        unwrap_angle(solution1[1], last_euler[1]),
        unwrap_angle(solution1[2], last_euler[2])
    ])
    
    sol2_unwrapped = np.array([
        unwrap_angle(solution2[0], last_euler[0]),
        unwrap_angle(solution2[1], last_euler[1]),
        unwrap_angle(solution2[2], last_euler[2])
    ])

    # 计算哪个解的总距离更小
    # 使用加权距离，因为 beta 角的变化通常更重要
    # 这里为了简单，我们用非加权的欧几里得距离的平方
    dist1 = np.sum((sol1_unwrapped - last_euler)**2)
    dist2 = np.sum((sol2_unwrapped - last_euler)**2)

    if dist1 < dist2:
        return sol1_unwrapped
    else:
        return sol2_unwrapped

class GamepadReader:
    """手柄读取器类"""
    
    def __init__(self, device_path):
        self.device_path = device_path
        self.file = None
        self.axes = [0] * 8  # 假设最多8个轴
        self.buttons = [0] * 100  # 假设最多100个按键
        self.previous_buttons = [0] * 100  # 上一帧的按键状态
        self.connected = False
        
    def connect(self):
        """连接手柄设备"""
        try:
            self.file = open(self.device_path, 'rb')
            # 设置文件为非阻塞模式
            fd = self.file.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            self.connected = True
            return True
        except (FileNotFoundError, PermissionError):
            self.connected = False
            return False
    
    def disconnect(self):
        """断开手柄连接"""
        if self.file:
            self.file.close()
            self.file = None
        self.connected = False
    
    def read_events(self):
        """读取手柄事件，返回按键按下事件列表"""
        if not self.connected or not self.file:
            return []
        
        button_presses = []
        
        try:
            # 检查是否有数据可读
            ready, _, _ = select.select([self.file], [], [], 0)
            if not ready:
                return button_presses
            
            # 限制读取次数，避免死循环
            max_events = 100  # 每次最多处理100个事件
            event_count = 0
            
            while event_count < max_events:
                try:
                    # 再次检查是否有数据可读（防止select的竞态条件）
                    ready, _, _ = select.select([self.file], [], [], 0)
                    if not ready:
                        break
                    
                    # 读取8字节的joystick事件
                    data = self.file.read(8)
                    if len(data) != 8:
                        break
                    
                    # 解析事件：time(4字节), value(2字节), type(1字节), number(1字节)
                    time, value, event_type, number = struct.unpack('<IhBB', data)
                    
                    if event_type & 0x01:  # 按键事件
                        # print(f"Button event: time={time}, value={value}, type={event_type}, number={number}")
                        if number < len(self.buttons):
                            # 检测按键按下（从0变为1）
                            if value == 1 and self.buttons[number] == 0:
                                button_presses.append(number)
                            
                            self.previous_buttons[number] = self.buttons[number]
                            self.buttons[number] = value

                
                    elif event_type & 0x02:  # 轴事件
                        if number < len(self.axes):
                            self.axes[number] = value
                    
                    event_count += 1
                
                except struct.error:
                    break
                except (BlockingIOError, OSError):
                    # 没有更多数据可读
                    break
                except Exception as e:
                    # 其他错误
                    break
        
        except Exception as e:
            # 连接断开或其他错误
            self.connected = False
        
        return button_presses
    
    def get_axis(self, axis_num):
        """获取轴值"""
        if axis_num < len(self.axes):
            return self.axes[axis_num]
        return 0
    
    def is_button_pressed(self, button_num):
        """检查按键是否被按下"""
        if button_num < len(self.buttons):
            return self.buttons[button_num] == 1
        return False

class WristControlNode(Node):
    """ROS2节点，用于接收VMC数据并发布机械臂关节位置"""
    
    def __init__(self):
        super().__init__('wrist_control_node')
        
        # self.declare_parameter('/topic_prefix', 'default_value')
        # self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        # print('topic_prefix:', self.topic_prefix)

        # 创建发布器
        self.qpos_publisher = self.create_publisher(
            usrMsg.ArmActuatorCmds, 
            # self.topic_prefix+'arm_actuators_cmds', 
            "/hardware/arm_actuators_cmds",
            10
        )
        
        # 初始化VMC数据
        self.pos_upper_arm = [0, 0, 0]
        self.rot_upper_arm = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        self.pos_lower_arm = [0, 0, 0]
        self.rot_lower_arm = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        self.pos_hand = [0, 0, 0]
        self.rot_hand = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        self.rot_upper_arm_left = [0, 0, 0, 1]  # 左上臂的四元数
        self.rot_lower_arm_left = [0, 0, 0, 1]
        self.rot_hand_left = [0, 0, 0, 1]  # 左手的四元数

        self.rot_foot = [0, 0, 0, 1]  # 右脚的四元数
        self.rot_foot_left = [0, 0, 0, 1]
        
        # 存储上一帧的欧拉角，用于跳变检测
        self.previous_euler = np.zeros(3)
        self.previous_euler2 = np.zeros(3)
        
        # 多圈计数 - 存储每个关节的累积位置
        self.cumulative_qpos = np.zeros(9)
        self.previous_qpos = np.zeros(9)
        self.is_first_frame = True
        
        # 控制标志
        self.listening = True
        
        # 初始化OSC
        self.osc = OSC(backend)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 创建定时器来处理数据和发布
        self.timer = self.create_timer(0.01, self.process_and_publish)  # 100Hz
        
        # 启动VMC接收
        self.start_vmc_receiver()
        
        self.get_logger().info('WristControlNode 已启动，开始接收VMC数据并发布qpos')

        self.gripper_l = 0.0  # 初始化夹爪位置
        self.gripper_r = 0.0  # 初始化夹爪位置
        self.gripper_l_real = 0.0
        self.gripper_r_real = 0.0  # 实际夹爪位置
        
        # 手柄相关初始化
        self.gamepads = []
        self.left_gamepad = None   # 左手手柄
        self.right_gamepad = None  # 右手手柄
        self.teleoperation_active = False  # 遥操作状态

        self.last_qpos=None

        self.last_euler = np.zeros(6)  # 上一帧的欧拉角
        # 初始化手柄
        self.init_gamepads()
        
        #缓启动
        self.start_time = -1.0

        # keyboard.Listener(
        #     on_press=self.on_key_press,
        #     on_release=self.on_key_release
        # ).start()
    def on_key_press(self, key):
        """按键按下事件处理"""
        try:
            if key.char == 'a':
                self.gripper_l += 2.5
                self.gripper_r += 2.5
                self.get_logger().info(f"夹爪位置增加: {self.gripper_l}")
            elif key.char == 'd':
                self.gripper_l -= 2.5
                self.gripper_r -= 2.5
                self.get_logger().info(f"夹爪位置减少: {self.gripper_l}")
        except AttributeError:
            pass
    def on_key_release(self, key):
        pass
    
    def init_gamepads(self):
        """初始化手柄设备"""
        self.get_logger().info("正在初始化手柄设备...")
        
        for device_path in GAMEPAD_DEVICES:
            gamepad = GamepadReader(device_path)
            if gamepad.connect():
                self.gamepads.append(gamepad)
                self.get_logger().info(f"手柄已连接: {device_path}")
            else:
                self.get_logger().warning(f"无法连接手柄: {device_path}")
        
        if len(self.gamepads) >= 2:
            self.get_logger().info(f"成功连接 {len(self.gamepads)} 个手柄")
        else:
            self.get_logger().warning(f"只连接了 {len(self.gamepads)} 个手柄，需要至少2个手柄")
    
    def identify_gamepads(self):
        """识别左右手手柄"""
        
        if len(self.gamepads) < 2:
            return
        
        for gamepad in self.gamepads:
            gamepad.read_events()  # 更新手柄状态
            
            axis4_value = gamepad.get_axis(AXIS_4)  # 上下轴
            axis5_value = gamepad.get_axis(AXIS_5)  # 左右轴
            
            # 检查摇杆是否向左（axis5为正值且超过阈值）
            if axis5_value > STICK_THRESHOLD and self.left_gamepad is None:
                self.left_gamepad = gamepad
                self.get_logger().info(f"识别左手手柄: {gamepad.device_path}")
            
            # 检查摇杆是否向右（axis5为负值且超过阈值）
            elif axis5_value < -STICK_THRESHOLD and self.right_gamepad is None:
                self.right_gamepad = gamepad
                self.get_logger().info(f"识别右手手柄: {gamepad.device_path}")
    
    def check_teleoperation_trigger(self):
        """检查遥操作触发条件"""
        if self.left_gamepad is None or self.right_gamepad is None:
            self.identify_gamepads()
            return
        
        # 读取手柄事件
        left_events = self.left_gamepad.read_events()
        right_events = self.right_gamepad.read_events()
        
        left_axis5 = self.left_gamepad.get_axis(AXIS_5)   # 左手手柄左右轴
        right_axis5 = self.right_gamepad.get_axis(AXIS_5) # 右手手柄左右轴
        
        # 检查启动条件：左手向左，右手向右
        if (left_axis5 > STICK_THRESHOLD and right_axis5 < -STICK_THRESHOLD):
            if not self.teleoperation_active:
                self.teleoperation_active = True
                self.start_time = time.time()  # 记录启动时间
                self.get_logger().info("遥操作已启动！")
        
        # 检查停止条件：左手向右，右手向左
        elif (left_axis5 < -STICK_THRESHOLD and right_axis5 > STICK_THRESHOLD):
            if self.teleoperation_active:
                self.teleoperation_active = False
                self.get_logger().info("遥操作已停止！")
        
        # 处理夹爪按键
        for button in left_events:
            if button == BUTTON_0:
                self.gripper_l += GRIPPER_STEP
                self.get_logger().info(f"左手按键0：夹爪位置增加 {self.gripper_l:.2f}")
            elif button == BUTTON_4:
                self.gripper_l -= GRIPPER_STEP
                self.get_logger().info(f"左手按键4：夹爪位置减少 {self.gripper_l:.2f}")
        
        for button in right_events:
            if button == BUTTON_0:
                self.gripper_r += GRIPPER_STEP
                self.get_logger().info(f"右手按键0：夹爪位置增加 {self.gripper_r:.2f}")
            elif button == BUTTON_4:
                self.gripper_r -= GRIPPER_STEP
                self.get_logger().info(f"右手按键4：夹爪位置减少 {self.gripper_r:.2f}")
        
        # 限制夹爪范围
        self.gripper_l = np.clip(self.gripper_l, -5.0, 0.0)
        self.gripper_r = np.clip(self.gripper_r, -5.0, 0.0)
    
    def signal_handler(self, sig, frame):
        """优雅地处理Ctrl+C信号"""
        self.get_logger().info("收到中断信号，正在停止...")
        self.listening = False
        self.osc.close()
        rclpy.shutdown()
        sys.exit(0)
    
    def received(self, event: Event):
        """VMC事件接收回调函数"""
        if isinstance(event, BoneTransformEvent):
            # 过滤右手相关骨骼
            if event.joint == Bone.RIGHT_UPPER_ARM:
                self.pos_upper_arm = [event.position.x, event.position.y, event.position.z]
                self.rot_upper_arm = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.RIGHT_LOWER_ARM:
                self.pos_lower_arm = [event.position.x, event.position.y, event.position.z]
                self.rot_lower_arm = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.RIGHT_HAND:
                self.pos_hand = [event.position.x, event.position.y, event.position.z]
                self.rot_hand = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.LEFT_UPPER_ARM:
                self.rot_upper_arm_left = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.LEFT_LOWER_ARM:
                self.rot_lower_arm_left = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.LEFT_HAND:
                self.rot_hand_left = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]

            elif event.joint == Bone.RIGHT_FOOT:
                self.rot_foot = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.LEFT_FOOT:
                self.rot_foot_left = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
    
    def start_vmc_receiver(self):
        """启动VMC接收器"""
        try:
            self.osc.open()
            # 创建接收器
            self.receiver = self.osc.create_receiver("0.0.0.0", 39539, "receiver1").open()
            on_receive(self.receiver, RootTransformEvent, self.received)
            on_receive(self.receiver, BoneTransformEvent, self.received)
            on_receive(self.receiver, DeviceTransformEvent, self.received)
            on_receive(self.receiver, BlendShapeEvent, self.received)
            on_receive(self.receiver, BlendShapeApplyEvent, self.received)
            on_receive(self.receiver, StateEvent, self.received)
            on_receive(self.receiver, RelativeTimeEvent, self.received)
            
            # 在单独线程中运行OSC
            self.osc_thread = threading.Thread(target=self.run_osc, daemon=True)
            self.osc_thread.start()
            
        except Exception as e:
            self.get_logger().error(f"启动VMC接收器失败: {e}")
    
    def run_osc(self):
        """在单独线程中运行OSC"""
        while self.listening:
            try:
                self.osc.run()
                time.sleep(0.001)  # 1ms延迟
            except Exception as e:
                if self.listening:  # 只有在还在监听时才记录错误
                    self.get_logger().error(f"OSC运行错误: {e}")
                break
    
    def calculate_qpos(self):
        """计算机械臂关节位置（支持多圈计数）"""
        qpos = np.zeros(16,dtype=np.float32,)  # 初始化qpos数组
        
        # 获取当前欧拉角 - 上臂
        # current_euler = quaternion_to_euler(self.rot_upper_arm, seq='XYX')
        # corrected_euler = current_euler
        
        corrected_euler = quaternion_to_xyx_euler_continuous(
            self.rot_upper_arm, 
            last_euler=self.previous_euler
        )

        # 更新控制器 - 上臂
        qpos[0+5] = corrected_euler[0] - radians(90)#+ radians(90)
        qpos[1+5] = -corrected_euler[1] - radians(90)
        qpos[2+5] = corrected_euler[2] + radians(90)  - radians(270) + radians(90) + radians(90) + radians(90)  #+ radians(45)
    

        # 更新上一帧的角度
        self.previous_euler = corrected_euler.copy()
        

        # 获取小臂绕Z轴的旋转分量
        lower_arm_euler = quaternion_to_euler(self.rot_lower_arm, seq='ZXZ')
        
        lower_arm_z_rotation = lower_arm_euler[0] + radians(90)  # 取Z轴旋转并取反
        
        # 获取手部旋转的欧拉角
        hand_euler = quaternion_to_euler(self.rot_hand, seq='XYZ')
        
        # 将小臂的Z轴旋转只应用到手部的X轴旋转上
        corrected_euler = hand_euler.copy()
        corrected_euler[0] = hand_euler[0] + lower_arm_z_rotation  # 只修改X轴旋转
        
        # 更新控制器
        # qpos[4] = (corrected_euler[0] + radians(90))
        qpos[4+5] = corrected_euler[0]
        qpos[5+5+3] = -corrected_euler[1]
        qpos[6+5+3] = -corrected_euler[2]

        # 更新上一帧的角度
        # self.previous_euler2 = corrected_euler.copy()
        
        # 手肘夹角
        q = np.array(self.rot_lower_arm)
        q = q / np.linalg.norm(q)
        
        x, y, z, w = q
        
        # 计算旋转后的X轴
        x_new_x = 1 - 2*y**2 - 2*z**2
        x_new_y = 2*x*y + 2*z*w
        x_new_z = 2*x*z - 2*y*w
        
        x_new = np.array([x_new_x, x_new_y, x_new_z])
        x_orig = np.array([1, 0, 0])
        
        # 计算点积和夹角
        dot_product = np.dot(x_orig, x_new)
        elbow_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 添加clip防止数值错误
        qpos[3+5] = -elbow_angle

        # left===============================================================
        # current_euler = quaternion_to_euler(self.rot_upper_arm_left, seq='XYX')
        # corrected_euler = current_euler
        corrected_euler = quaternion_to_xyx_euler_continuous(
            self.rot_upper_arm_left, 
            last_euler=self.previous_euler2
        )

        
        # 更新控制器 - 上臂
        qpos[0] = corrected_euler[0] - radians(90)
        qpos[1] = -(corrected_euler[1] - radians(90)) 
        qpos[2] = -corrected_euler[2] - radians(45)
        
        # 更新上一帧的角度
        self.previous_euler2 = corrected_euler.copy()
        

        # 获取小臂绕Z轴的旋转分量
        lower_arm_euler = quaternion_to_euler(self.rot_lower_arm_left, seq='ZXZ')
        
        lower_arm_z_rotation = lower_arm_euler[0] + radians(90)  # 取Z轴旋转并取反
        
        # 获取手部旋转的欧拉角
        hand_euler = quaternion_to_euler(self.rot_hand_left, seq='XYZ')
        
        # 将小臂的Z轴旋转只应用到手部的X轴旋转上
        corrected_euler = hand_euler.copy()
        corrected_euler[0] = -hand_euler[0] + lower_arm_z_rotation  # 只修改X轴旋转
        
        # 更新控制器
        # qpos[4] = (corrected_euler[0] + radians(90))
        qpos[4] = corrected_euler[0] + radians(180)
        qpos[5+5] = corrected_euler[1]
        qpos[6+5] = -corrected_euler[2]

        # 更新上一帧的角度
        # self.previous_euler2 = corrected_euler.copy()
        
        # 手肘夹角
        q = np.array(self.rot_lower_arm_left)
        
        q = q / np.linalg.norm(q)
        
        x, y, z, w = q
        
        # 计算旋转后的X轴
        x_new_x = 1 - 2*y**2 - 2*z**2
        x_new_y = 2*x*y + 2*z*w
        x_new_z = 2*x*z - 2*y*w
        
        x_new = np.array([x_new_x, x_new_y, x_new_z])
        x_orig = np.array([1, 0, 0])
        
        # 计算点积和夹角
        dot_product = np.dot(x_orig, x_new)
        elbow_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 添加clip防止数值错误
        qpos[3] = -elbow_angle
        
        for i in range(16):
            qpos[i] = limit_angle_range(qpos[i])

        qpos[2] = np.clip(qpos[2],radians(-150), radians(150))  # 限制肩关节Z轴旋转在-180到180度之间
        qpos[2+5] = np.clip(qpos[2+5],radians(-150), radians(150))  # 限制肩关节Z轴旋转在-180到180度之间
        
        # self.gripper=np.clip(self.gripper, -5.0, 0.0)
        # self.gripper_real = 0.95*self.gripper_real+0.05*self.gripper  # 限制夹爪位置在0到1之间 

        # qpos[12]=np.clip(quaternion_to_euler(self.rot_foot_left, seq='XYZ',degrees_output=True)[0]/20.0*5, -5.0, 0.0)  # 限制夹爪位置在-5到0之间
        # qpos[15]=np.clip(quaternion_to_euler(self.rot_foot, seq='XYZ',degrees_output=True)[0]/20.0*5, -5.0, 0.0)

        # 使用手柄控制的夹爪位置
        # qpos[12] = self.gripper_l  # 左手夹爪
        self.gripper_l_real = 0.95*self.gripper_l_real+0.05*self.gripper_l 
        # qpos[15] = self.gripper_r  # 右手夹爪
        self.gripper_r_real = 0.95*self.gripper_r_real+0.05*self.gripper_r 
        qpos[12] = self.gripper_l_real
        qpos[15] = self.gripper_r_real

        if self.last_qpos is None:
            self.last_qpos = qpos.copy()
        
        dof_s_limit = 0.01
        for i in range(16):
            if abs(qpos[i]-self.last_qpos[i])>dof_s_limit:
                qpos[i]=self.last_qpos[i]+np.sign(qpos[i]-self.last_qpos[i])*dof_s_limit
        
        self.last_qpos = qpos.copy()

        # qpos[12] = np.clip(self.gripper_l_real, -5.0, 0.0)  # 左手夹爪
        # qpos[15] = np.clip(self.gripper_r_real, -5.0, 0.0)  # 右手夹爪

        # qpos_mask = [0,0,0,0,0,
        #              1,1,1,1,1,
        #              0,0,0,
        #              1,1,1]
        # qpos_mask = [1,1,1,1,1,
        #              0,0,0,0,0,
        #              1,1,1,
        #              0,0,0]
        qpos_mask = [1,1,1,1,1,
                     1,1,1,1,1,
                     1,1,1,
                     1,1,1,]
        # qpos_mask = [0,0,0,0,0,
        #              0,0,0,0,0,
        #              0,0,1,
        #              0,0,1]
        # qpos_mask = [0,0,0,0,0,
        #             0,0,0,0,0,
        #             0,0,0,
        #             0,0,0]
        qpos = qpos * qpos_mask
        # # 多圈计数处理
        # if self.is_first_frame:
        #     # 第一帧直接设置初始值
        #     self.cumulative_qpos = qpos.copy()
        #     self.previous_qpos = qpos.copy()
        #     self.is_first_frame = False
        # else:
        #     # 计算每个关节的角度差值
        #     for i in range(len(qpos)):
        #         angle_diff = qpos[i] - self.previous_qpos[i]
                
        #         # 处理角度跳变（从-π到π或从π到-π）
        #         if angle_diff > np.pi:
        #             angle_diff -= 2 * np.pi
        #         elif angle_diff < -np.pi:
        #             angle_diff += 2 * np.pi
                
        #         # 累积角度差值到总角度
        #         self.cumulative_qpos[i] += angle_diff
            
        #     # 更新上一帧的角度
        #     self.previous_qpos = qpos.copy()

        
        
        return qpos
    
    def process_and_publish(self):
        """处理数据并发布qpos"""
        try:
            # 检查手柄状态和遥操作触发
            self.check_teleoperation_trigger()
            
            # 计算qpos
            qpos = self.calculate_qpos()
            
            # 如果遥操作未激活，可以选择不发布或发布默认位置
            if not self.teleoperation_active:
                # 可以选择发布默认姿态或者不发布
                # qpos = np.zeros(16)  # 或者使用默认姿态
                return
            
            # 创建消息
            msg = usrMsg.ArmActuatorCmds()
            #    arm_pos = msg.pos
            #   arm_vel = msg.vel
            #   arm_kp = msg.kp
            #    arm_kd = msg.arm_kd
            msg.kp = joint_kp[-16:].tolist()  # 将numpy数组转换为列表

            soft_k = np.clip((time.time()-self.start_time)/3.0,0.1,1.0)
            soft_kp = joint_kp[-16:] * soft_k
            # kp_mask = [0,0,0,0,0,
            #            0,0,0,0,0,
            #            0,0,0,
            #            0,0,0]
            # kp_mask = [1,1,1,1,1,
            #            0,0,0,0,0,
            #            1,1,1,
            #            0,0,0]
            # kp_mask = [0,0,0,0,0,
            #            1,1,1,1,1,
            #            0,0,0,
            #            1,1,1]
            kp_mask = [1,1,1,1,1,
                       1,1,1,1,1,
                       1,1,1,
                       1,1,1]
            msg.kp = (soft_kp * kp_mask).tolist()  # 将numpy数组转换为列表
            msg.kd = joint_kd[-16:].tolist()  # 将numpy数组转换为列表
            # msg.kp = np.zeros_like(qpos).tolist()
            msg.pos = qpos.tolist()  # 将numpy数组转换为列表
            msg.vel = np.zeros_like(qpos).tolist()  # 初始化速度为0
            # msg.data = qpos.tolist()
            
            # 发布消息
            self.qpos_publisher.publish(msg)
            
            # 打印调试信息（可选）
            # self.get_logger().info(f"发布qpos: {qpos}")
            
        except Exception as e:
            self.get_logger().error(f"处理和发布数据时出错: {e}")
    
    def destroy_node(self):
        """销毁节点时的清理工作"""
        self.listening = False
        if hasattr(self, 'osc'):
            self.osc.close()
        
        # 清理手柄资源
        for gamepad in self.gamepads:
            gamepad.disconnect()
        
        super().destroy_node()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = WristControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
