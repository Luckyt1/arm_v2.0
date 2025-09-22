import os
import mujoco
import mujoco.viewer
import numpy as np
import math
import time

# The inverse_kinematics function provided by the user is for a 2-DOF arm.
# For a 7-DOF arm, a more advanced inverse kinematics solution is required.
# This typically involves numerical methods (e.g., Jacobian-based inverse kinematics)
# or dedicated robotics libraries.
# For demonstration, this script will focus on loading the 7-DOF arm model
# and applying direct joint control.
# If a 7-DOF IK solution were available, it would compute target joint angles
# which would then be used to drive the robot.
def inverse_kinematics_2dof(x, z, L1, L2, base_height=1.2):
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
    z_rel = z - base_height
    D = math.sqrt(x**2 + z_rel**2)
    
    cos_theta2 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = math.acos(min(1, max(cos_theta2, -1)))
    
    sin_theta2 = math.sqrt(1 - cos_theta2**2)
    theta1 = math.atan2(z_rel, x) - math.atan2(L2 * sin_theta2, L1 + L2 * cos_theta2)

    return theta1, theta2

def main():
    # Construct the path to the XML model
    model_path = os.path.join(os.path.dirname(__file__), 'robot_arm.xml')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the Mujoco model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Define target joint positions (example: all joints at 0)
    # For a 7-DOF arm, you would have 7 target joint angles.
    # In a real application, these would come from an IK solver.
    target_joint_positions = np.zeros(model.nu) # nu is number of actuators/controllable joints

    # Simple P-controller gains
    kp = 50.0 # 降低增益以提高稳定性
    kv = 5.0  # 降低增益以提高稳定性

    print("Starting 7-DOF arm control simulation. Press ESC to exit.")
    print("Note: This script demonstrates direct joint control. For end-effector control,")
    print("a 7-DOF inverse kinematics solver would be needed.")

    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute control torques using a simple PD controller
            # This moves the joints towards the target_joint_positions
            joint_errors = target_joint_positions - data.qpos
            joint_velocities = data.qvel
            
            # Apply proportional and derivative control
            data.ctrl = kp * joint_errors - kv * joint_velocities

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Optional: Add a small delay to slow down the simulation if needed
            time.sleep(0.01)
    print("Simulation ended.")

if __name__ == "__main__":
    main()
