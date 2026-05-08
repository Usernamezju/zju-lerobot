"""快速测试：机械臂能否正常运动（正运动学 + 可视化）"""
import pinocchio as pin
import numpy as np
import time

from pinocchio.visualize import MeshcatVisualizer

urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_path, package_dirs=[mesh_dir]
)
data = model.createData()

print(f"关节数量: {model.nq}")
print(f"关节名称: {list(model.names)}")
print(f"关节限位 (lower): {model.lowerPositionLimit}")
print(f"关节限位 (upper): {model.upperPositionLimit}")
print(f"neutral: {pin.neutral(model)}")

# 可视化
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

print("\n===== 测试 1: Neutral 姿态 =====")
q = pin.neutral(model)
viz.display(q)
print(f"关节角: {q}")
print(f"浏览器查看: http://127.0.0.1:7000/static/")
input("按 Enter 继续...")

print("\n===== 测试 2: 单关节正弦运动 =====")
q = pin.neutral(model)
for t in np.linspace(0, 2*np.pi, 200):
    q[0] = np.sin(t) * 0.5   # 关节 1 摆动
    q[2] = np.sin(t * 1.5) * 0.3  # 关节 3 摆动
    viz.display(q)
    time.sleep(0.05)
print("单关节运动完成 ✓")
input("按 Enter 继续...")

print("\n===== 测试 3: 末端轨迹（正运动学验证）=====")
q = pin.neutral(model)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

ee_id = model.getFrameId("gripper_frame_link")
pos0 = data.oMf[ee_id].translation.copy()
print(f"neutral 位姿下末端位置: {pos0}")

# 规划一组关节角，让末端画一条竖直线
for z_offset in np.linspace(0, -0.15, 100):
    q[0] = z_offset * 3
    q[2] = -z_offset * 2
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pos = data.oMf[ee_id].translation
    print(f"  末端位置: {pos}", end="\r")
    viz.display(q)
    time.sleep(0.03)
print("\n末端运动测试完成 ✓")

print("\n===== 测试完成 =====")
print("机械臂可以正常运动！")
print("\n如果上述测试失败，请检查:")
print("  1. Meshcat 浏览器页面是否打开 (http://127.0.0.1:7000/static/)")
print("  2. URDF 文件是否正确")
print("  3. 是否有其他 pinocchio 相关报错")
