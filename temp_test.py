# temp_test.py
from kinematics import Kinematics
import numpy as np

urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"

kin = Kinematics(urdf_path, mesh_dir)

# 只约束位置，不约束姿态
target_pos = np.array([0.3, 0.0, 0.1])
q_sol, success = kin.inverse_kinematics(target_pos, np.eye(3))
print(f"成功: {success}")
print(f"关节角: {q_sol}")

# 验证
pos, _ = kin.forward_kinematics(q_sol)
print(f"实际末端位置: {pos}")
print(f"误差: {np.linalg.norm(pos - target_pos):.6f}")