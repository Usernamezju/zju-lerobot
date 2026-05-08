"""
SO-ARM100 工作空间检测工具
随机采样关节角 → 正运动学映射末端位置 → 分析可达范围
"""

import numpy as np
import pinocchio as pin
import sys

def check_workspace(urdf_path, mesh_dir, n_samples=50000):
    """随机采样关节角，计算末端执行器可达位置"""
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_dir])
    data = model.createData()
    ee_id = model.getFrameId("gripper_frame_link")

    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit

    print(f"关节数: {model.nq}")
    print(f"采样点数: {n_samples}")
    print()

    positions = []
    for i in range(n_samples):
        q = np.random.uniform(lower, upper)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pos = data.oMf[ee_id].translation.copy()
        positions.append(pos)

    positions = np.array(positions)

    # 计算工作空间统计
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    print("=" * 50)
    print("工作空间范围 (mm)")
    print("=" * 50)
    print(f"X 轴: [{x_min*1000:6.1f}, {x_max*1000:6.1f}] mm")
    print(f"Y 轴: [{y_min*1000:6.1f}, {y_max*1000:6.1f}] mm")
    print(f"Z 轴: [{z_min*1000:6.1f}, {z_max*1000:6.1f}] mm")
    print()

    # 在 XY 切面上看不同高度的工作范围
    print("=" * 50)
    print("各高度层 XY 可达范围")
    print("=" * 50)
    for z_level in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        mask = np.abs(positions[:, 2] - z_level) < 0.02
        if mask.sum() > 0:
            pts = positions[mask]
            print(f"z={z_level:.2f}m:  X [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]  "
                  f"Y [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]  (采样 {mask.sum()} 点)")

    return positions


def check_reachability(urdf_path, mesh_dir, target_pos, n_init=50, max_iter=1000, eps=1e-4):
    """
    从多个随机初始猜测出发尝试 IK，判断目标是否可达。
    返回: (success_rate, best_error, solutions)
    """
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_dir])
    data = model.createData()
    ee_id = model.getFrameId("gripper_frame_link")
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit

    target_pos = np.asarray(target_pos)
    target_rot = np.eye(3)
    target_SE3 = pin.SE3(target_rot, target_pos)

    successes = 0
    best_error = float("inf")
    solutions = []

    for i in range(n_init):
        q = np.random.uniform(lower, upper)

        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current_SE3 = data.oMf[ee_id]
            error = pin.log6(current_SE3.inverse() * target_SE3).vector
            err_norm = np.linalg.norm(error)

            if err_norm < eps:
                successes += 1
                solutions.append(q.copy())
                break

            J = pin.computeFrameJacobian(
                model, data, q, ee_id, pin.ReferenceFrame.LOCAL
            )
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ error
            q = pin.integrate(model, q, dq)
        else:
            # 最后一次迭代后的误差
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current_SE3 = data.oMf[ee_id]
            error = pin.log6(current_SE3.inverse() * target_SE3).vector
            err_norm = np.linalg.norm(error)
            if err_norm < best_error:
                best_error = err_norm

    success_rate = successes / n_init
    return success_rate, best_error, solutions


if __name__ == "__main__":
    urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"

    print("=" * 50)
    print("SO-ARM100 工作空间检测")
    print("=" * 50)
    print()

    # 1. 随机采样分析工作空间
    positions = check_workspace(urdf_path, mesh_dir, n_samples=50000)

    # 2. 检查 IK 报错的目标点
    print()
    print("=" * 50)
    print("目标点 IK 可达性检测 (50 次随机初始猜测)")
    print("=" * 50)
    target = [0.3, 0.0, 0.1]
    rate, err, sols = check_reachability(urdf_path, mesh_dir, target, n_init=50)
    print(f"目标 ({target[0]}, {target[1]}, {target[2]})")
    print(f"  IK 成功率: {rate*100:.1f}%")
    print(f"  最佳残余误差: {err:.6f}")
    if rate == 0:
        print("  → 结论: 该点不可达 (IK 无法收敛)")

    # 3. 再试一些候选点
    print()
    print("=" * 50)
    print("几个候选点的 IK 可达性")
    print("=" * 50)
    test_points = [
        [0.2, 0.0, 0.05],
        [0.2, 0.0, 0.15],
        [0.25, 0.0, 0.1],
        [0.15, 0.1, 0.1],
        [0.2, -0.1, 0.15],
        [0.3, 0.0, 0.2],
        [0.1, 0.0, 0.05],
    ]
    for pt in test_points:
        rate, err, _ = check_reachability(urdf_path, mesh_dir, pt, n_init=20)
        status = "可达" if rate > 0 else "不可达"
        print(f"  ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})  {status}  (成功率 {rate*100:.0f}%, 残差 {err:.6f})")
