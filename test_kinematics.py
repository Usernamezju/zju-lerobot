# test_kinematics.py
import numpy as np
import pinocchio as pin
from kinematics import Kinematics

urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"

kin = Kinematics(urdf_path, mesh_dir)
for i, frame in enumerate(kin.model.frames):
    print(f"{i}: {frame.name}")

# 测试正运动学
q = pin.neutral(kin.model)
pos, rot = kin.forward_kinematics(q)
print(f"正运动学结果:")
print(f"  末端位置: {pos}")
print(f"  末端旋转:\n{rot}")

# 测试逆运动学
target_pos = np.array([0.2, 0.0, 0.3])
target_rot = np.eye(3)
q_sol, success = kin.inverse_kinematics(target_pos, target_rot)
print(f"\n逆运动学结果:")
print(f"  成功: {success}")
print(f"  关节角: {q_sol}")

# 验证：用正运动学验证IK结果
pos_check, _ = kin.forward_kinematics(q_sol)
print(f"  验证末端位置: {pos_check}")
print(f"  误差: {np.linalg.norm(pos_check - target_pos):.6f}")