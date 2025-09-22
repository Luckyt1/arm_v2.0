import mujoco
import os
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from first import PIDController
# 定义XML字符串，用于定义机器人模型
XML=r"""
<mujoco model="arm.SLDASM">
  <compiler angle="radian" autolimits="true"/>
  <option>
    <flag gravity="disable"/>
  </option>
  <statistic meansize="0.103375" extent="0.820674" center="-0.000696085 0.000716649 0.342986"/>
  
  <default>
    <joint damping="1" stiffness="10"/>
    <position kp="100"/>
  </default>
  
  <asset>
    <mesh name="base_link" file="base_link.STL"/>
    <mesh name="link1" file="link1.STL"/>
    <mesh name="link2" file="link2.STL"/>
    <mesh name="link3" file="link3.STL"/>
    <mesh name="link4" file="link4.STL"/>
    <mesh name="link5" file="link5.STL"/>
    <mesh name="link6" file="link6.STL"/>
    <mesh name="link7" file="link7.STL"/>
    <mesh name="gripper_l" file="gripper_l.STL"/>
    <mesh name="gripper_r" file="gripper_r.STL"/>
  </asset>  <worldbody>
    <geom type="mesh" rgba="1 1 1 1" mesh="base_link"/>
    <body name="link1">
      <inertial pos="-1.01502e-06 -0.00315105 0.0431262" quat="0.704635 0.0582641 -0.038985 0.706098" mass="0.363576" diaginertia="9.56788e-05 7.78423e-05 7.69165e-05"/>
      <joint name="joint_link1" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
      <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link1"/>
      <body name="link2" pos="0 0 0.048" quat="0.707105 0.707108 0 0">
        <inertial pos="4.84416e-05 0.060778 1.93033e-05" quat="0.709402 -0.00153027 0.00354105 0.704794" mass="0.345274" diaginertia="8.8169e-05 8.23659e-05 7.40042e-05"/>
        <joint name="joint_link2" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
        <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link2"/>
        <body name="link3" pos="0 0.094 0" quat="0.707105 -0.707108 0 0">
          <inertial pos="0.00231671 1.99517e-05 0.123131" quat="0.999896 0.000413564 0.0144106 0.000241483" mass="0.420046" diaginertia="0.000125858 0.00011213 8.58521e-05"/>
          <joint name="joint_link3" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
          <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link3"/>
          <body name="link4" pos="0 0 0.146" quat="0.707107 0 0.707107 0">
            <inertial pos="-0.0787171 0.000283571 -0.000118132" quat="0.487732 0.504794 0.51249 0.494626" mass="0.530694" diaginertia="0.000349053 0.000264016 0.000200715"/>
            <joint name="joint_link4" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
            <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link4"/>
            <body name="link5" pos="-0.1275 0 0" quat="0.707107 0 -0.707107 0">
              <inertial pos="-0.000555169 -1.36007e-08 0.0129923" quat="0.482896 0.516538 0.516538 0.482896" mass="0.1001848" diaginertia="4.66972e-05 3.42929e-05 2.25957e-05"/>
              <joint name="joint_link5" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
              <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link5"/>
              <body name="link6" pos="0 0 0.0415" quat="0.707107 0 0.707107 0">
                <inertial pos="-0.0296928 0.00138012 0.000664094" quat="0.474571 0.554121 0.374767 0.572086" mass="0.617362" diaginertia="0.000188386 0.000187871 0.000129882"/>
                <joint name="joint_link6" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
                <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link6"/>
                <body name="link7" pos="-0.059 0 -0.0007" quat="-0.500002 0.5 0.5 0.499998">
                  <inertial pos="-2.75193e-06 -0.0837463 0.000611285" quat="0.710474 0.703513 0.0121989 -0.0121333" mass="0.607928" diaginertia="0.000564844 0.000382701 0.000264212"/>
                  <joint name="joint_link7" pos="0 0 0" axis="0 0 -1" range="-1.5708 1.5708"/>
                  <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="link7"/>
                  <body name="gripper_l" pos="0 -0.105 -0.0039975" quat="-3.67321e-06 1 0 0">
                    <inertial pos="-0.00323360 0.0218652 0.0128604" mass="0.1029366" diaginertia="1.82575e-05 1.47403e-05 1.08564e-05"/>
                    <joint name="joint_gripper_l" pos="0 0 0" axis="0 0 1" type="slide" range="0 0.05"/>
                    <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="gripper_l"/>
                  </body>
                  <body name="gripper_r" pos="0 -0.105 0.0053984" quat="-3.67321e-06 1 0 0">
                    <inertial pos="0.00323360 0.0218652 -0.0128604" mass="0.1029366" diaginertia="1.82575e-05 1.47403e-05 1.08564e-05"/>
                    <joint name="joint_gripper_r" pos="0 0 0" axis="0 0 -1" type="slide" range="0 0.05"/>
                    <geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="gripper_r"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <contact>
    <exclude body1="gripper_l" body2="gripper_r"/>
  </contact>
  
#   <actuator>
#     <position name="act_joint_link1" joint="joint_link1" kp="100" kv="0.5"/>
#     <position name="act_joint_link2" joint="joint_link2" kp="200" kv="5"/>
#     <position name="act_joint_link3" joint="joint_link3" kp="100" kv="0.5"/>
#     <position name="act_joint_link4" joint="joint_link4" kp="150" kv="2"/>
#     <position name="act_joint_link5" joint="joint_link5" kp="100" kv="0.5"/>
#     <position name="act_joint_link6" joint="joint_link6" kp="100" kv="0.5"/>
#     <position name="act_joint_link7" joint="joint_link7" kp="100" kv="0.5"/>
#     <position name="act_gripper_l" joint="joint_gripper_l" kp="100" ctrlrange="0 0.05" kv="5"/>
#     <position name="act_gripper_r" joint="joint_gripper_r" kp="100" ctrlrange="0 0.05" kv="5"/>
#   </actuator>
</mujoco>
"""
def inverse_kinematics(x, z, L1, L2, base_height=1.2):
    """
    根据末端点坐标在X-Z平面上计算两个关节的角度（逆解算）。
    
    参数:
    - x: 末端点的X坐标。
    - z: 末端点的Z坐标。
    - L1: 第一节长度。
    - L2: 第二节长度。
    - base_height: 基座的高度。
    
    返回:
    - theta1: 第一个关节的角度（弧度）。
    - theta2: 第二个关节的角度（弧度）。
    """
    # 计算末端点到基座的相对高度
    z_rel = z - base_height
    D = math.sqrt(x**2 + z_rel**2)  # 末端点到原点的距离在X-Z平面上计算
    
    # 使用余弦定理计算第二个关节的角度
    cos_theta2 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = math.acos(min(1, max(cos_theta2, -1)))  # 确保acos的参数在合法范围内
    
    # 计算第一个关节的角度
    sin_theta2 = math.sqrt(1 - cos_theta2**2)
    theta1 = math.atan2(z_rel, x) - math.atan2(L2 * sin_theta2, L1 + L2 * cos_theta2)

    # print('D=',D,'theta1=',theta1,'theta2=',theta2)

    return theta1, theta2
#最后返还两个关节的及角度

def trajectory_tracking(points, duration):
    """
    轨迹跟踪函数，控制机械臂依次沿多个点移动。
    points: 点列表，每个点为(x, z)格式。
    duration: 每段轨迹的持续时间。
    """
    for i in range(len(points) - 1):
        start_pos = points[i]
        end_pos = points[i + 1]


ASSETS=dict()
# 从文件中读取STL文件，并存储到ASSETS字典中
with open('/home/tang/mjcf/mjmodel.xml', 'rb') as f:
  ASSETS['mjmodel.stl'] = f.read()


# 创建MjData对象
model = mujoco.MjModel.from_xml_string(XML, ASSETS)
data = mujoco.MjData(model)


dt = model.opt.timestep
dof = model.nv
print("DoF of the model: "+str(dof))


# 初始化关节状态
data.qpos = np.zeros(dof)
data.qvel = np.zeros(dof)


#规划路径
T = 10.0
num_step = int(T/dt)
t = np.linspace(0.0,T,num_step + 1) # 修正：确保t的长度与num_step + 1一致，以避免广播错误
q_plan = np.zeros((dof,num_step + 1))
# print(np.sin(t))
q = np.zeros((dof, num_step + 1))
tau = np.zeros((dof, num_step + 1))

#初始化数组

for i in range(2):
    np.sin(t)
    print(np.sin(t))


# 确保kp、kd和tau_lim列表的长度至少与模型的自由度相同
kp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 示例：长度为9
kd = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 示例：长度为9
tau_lim = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 示例：长度为9

tang =np.zeros(2)
pid_controller = []
for i in range(dof):
    pid_controller.append(PIDController(kp=kp[i], kd=kd[i],ki=0, limit=tau_lim[i]))

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step <= num_step:
        step_start = time.time()
 
        tang=inverse_kinematics(0.1, 0.1, 0.2, 0.2, base_height=0.1)
        #运动学逆解算
        
        # for i in range(dof):
        #     tau[i, step] = pid_controller[i].update(data.qpos[i], data.qvel[i], q_plan[i,step])
        #     # print(data.qvel[1])
        #     data.ctrl[i] = tau[i, step]

        for i in range(2):
            data.ctrl[i]=tang[i]    
        mujoco.mj_step(model, data)

        # for i in range(dof):
        #     q[i, step] = data.qpos[i]
        
        viewer.sync()
        step = step+1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

