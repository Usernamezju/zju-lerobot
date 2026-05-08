"""
基座位置自动调整工具
搜索合适的机械臂基座偏移量，使所有目标点进入工作空间
"""

import numpy as np
import pinocchio as pin


def find_reachable_base(urdf_path, mesh_dir, target_positions,
                        search_range=0.3, step=0.05, n_init=5,
                        max_iter=500, eps=1e-4):
    """
    搜索机械臂基座偏移量，使得所有目标点可达。

    参数:
        target_positions: list of (x, y, z) 世界坐标系下的目标点
        search_range: 搜索范围 (米)，在以原点为中心的正方形区域内搜索
        step: 搜索步长 (米)
        n_init: 每个候选位置对每个目标点的 IK 随机尝试次数
    返回:
        base_offset: (3,) ndarray 基座在世界坐标系中的位置 (z=0)
        success: bool 是否找到可行位置
    """
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_dir])
    data = model.createData()
    ee_id = model.getFrameId("gripper_frame_link")
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit

    # ── 1. 生成候选基座偏移 (按距原点距离升序) ──
    half = int(search_range / step)
    candidates = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            candidates.append(np.array([dx * step, dy * step, 0.0]))
    candidates.sort(key=lambda o: np.linalg.norm(o))

    def _ik_reachable(target_pos):
        """对单个目标点做多次 IK 检测"""
        target_SE3 = pin.SE3(np.eye(3), np.asarray(target_pos))
        for _ in range(n_init):
            q = np.random.uniform(lower, upper)
            for _ in range(max_iter):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                error = pin.log6(data.oMf[ee_id].inverse() * target_SE3).vector
                if np.linalg.norm(error) < eps:
                    return True
                J = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL)
                q = pin.integrate(model, q, np.linalg.pinv(J) @ error)
        return False

    # ── 2. 先检测原始位置 (基座不动) ──
    print("检测原始位置是否可达...")
    all_reachable = True
    for pos in target_positions:
        if not _ik_reachable(pos):
            all_reachable = False
            print(f"  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) 不可达，开始搜索基座偏移")
            break
    if all_reachable:
        print("所有目标点在原始基座位置下可达，无需移动")
        return np.zeros(3), True

    # ── 3. 搜索基座偏移 ──
    print(f"搜索合适基座位置 (共 {len(candidates)} 个候选)...")
    for i, base_offset in enumerate(candidates):
        if (i + 1) % 50 == 0:
            print(f"  已搜索 {i+1}/{len(candidates)} 个位置...")

        all_reachable = True
        for pos in target_positions:
            robot_pos = np.array(pos) - base_offset
            if not _ik_reachable(robot_pos):
                all_reachable = False
                break

        if all_reachable:
            msg = f"找到可达基座位置: ({base_offset[0]:.3f}, {base_offset[1]:.3f}, 0)"
            print(f"\n{'=' * 50}")
            print(msg)
            print(f"{'=' * 50}")
            return base_offset, True

    print("未找到可使所有目标点可达的基座位置，使用原点 (0, 0, 0)")
    return np.zeros(3), False
