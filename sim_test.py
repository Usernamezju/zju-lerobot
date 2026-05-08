import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
from obstacle import ObstacleManager
from object import SceneObject
from trajectory import Trajectory
from check_workspace import check_workspace, check_reachability
from base_adjust import find_reachable_base
import time
import argparse

parser = argparse.ArgumentParser(description='SO-ARM100 路径规划仿真')
parser.add_argument('--method', type=str, default='quintic',
                    choices=['quintic', 'parabolic'],
                    help='插值方式: quintic(五次多项式) 或 parabolic(抛物线过渡)')
args = parser.parse_args()

urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"


model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_path,
    package_dirs=[mesh_dir]
)


viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

q = pin.neutral(model)
viz.display(q)

# ── 定义世界坐标系下的目标位置 ──
world_target_pos  = (0.300, 0.000, 0.300)
world_place_pos   = (0.100, 0.300, 0.300)
world_box_pos     = (0.2, 0.1, 0.2)
world_sphere_pos  = (0.3, -0.1, 0.3)
Place_box_size = [0.08, 0.08, 0.04]

# ── 自动调整机械臂基座位置，使目标点可达 ──
print("\n搜索合适的基座位置...")
base_offset, found = find_reachable_base(
    urdf_path, mesh_dir,
    target_positions=[world_target_pos, world_place_pos],
    search_range=0.3, step=0.05, n_init=5
)

if found and np.linalg.norm(base_offset) > 0:
    print(f"基座偏移至: ({base_offset[0]:.3f}, {base_offset[1]:.3f}, 0)")
else:
    print("基座保持原点 (0, 0, 0)")

# ── 将世界坐标转换到机器人坐标系 ──
def to_robot(pos):
    return (pos[0] - base_offset[0], pos[1] - base_offset[1], pos[2])

target_pos = to_robot(world_target_pos)
place_pos  = to_robot(world_place_pos)

print("\n目标点在机器人坐标系中的位置:")
print(f"  target:   ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
print(f"  place:    ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")

# ── 在机器人坐标系中添加障碍物和目标到可视化窗口 ──
obs = ObstacleManager(viz.viewer["obstacles"])
obs.add_box("box1", position=to_robot(world_box_pos), size=[0.01, 0.01, 0.01])
obs.add_sphere("sphere1", position=to_robot(world_sphere_pos), radius=0.03)

place_box_size = Place_box_size
scene = SceneObject(viz.viewer["objects"])
scene.add_target("target", position=target_pos)
scene.add_place_box("place_box", position=place_pos, size=place_box_size)

# ── 验证可达性 ──
print("\n验证目标点可达性...")
for label, pos in [("target", target_pos), ("place_box", place_pos)]:
    rate, err, _ = check_reachability(urdf_path, mesh_dir, pos, n_init=20)
    status = "可达" if rate > 0 else "不可达"
    print(f"  {label} ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  {status}  "
          f"(成功率 {rate*100:.0f}%, 残差 {err:.6f})")

# ── 路径规划 ──
traj = Trajectory(urdf_path, mesh_dir, obstacles=obs.get_all())
result = traj.plan(target_pos, place_box_pos=place_pos,
                   place_box_size=place_box_size,
                   method=args.method, n_rrt_trials=3)
if result is None:
    print("\n路径规划失败，请调整目标点位置")
    input("按 Enter 退出...")
    exit(1)
best_traj = result['trajectory']

# 执行运动
print("\n===== 正向运动（抓取 → 放置） =====")
for q_step in best_traj:
    viz.display(q_step)
    time.sleep(0.01)

# 夹爪全开保持 2 秒
print("释放物体，保持 2 秒...")
viz.display(best_traj[-1])
time.sleep(2.0)

# 原路返回
print("===== 原路返回 =====")
for q_step in reversed(best_traj):
    viz.display(q_step)
    time.sleep(0.01)

print("\n===== 完成 =====")


print("关节数量:", model.nq)
print("关节名称:", [model.names[i] for i in range(len(model.names))])
print("\n浏览器查看: http://127.0.0.1:7000/static/")
input("按 Enter 退出...")
