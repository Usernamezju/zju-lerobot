# trajectory.py
import numpy as np
import pinocchio as pin
from kinematics import Kinematics

class Trajectory:
    def __init__(self, urdf_path, mesh_dir, obstacles=None, steps_per_segment=100, safe_height=0.05):
        self.kin = Kinematics(urdf_path, mesh_dir)
        self.steps_per_segment = steps_per_segment
        self.safe_height = safe_height
        self.obstacles = obstacles if obstacles is not None else {}

    # ── 插值 ──
    def quintic_segment(self, q_start, q_end):
        """单段五次多项式插值"""
        dq = q_end - q_start
        s = np.linspace(0, 1, self.steps_per_segment)
        traj = q_start + dq * (10*s**3 - 15*s**4 + 6*s**5)[:, None]
        return traj

    def parabolic_segment(self, q_start, q_end, blend_ratio=0.3):
        """单段抛物线过渡插值"""
        dq = q_end - q_start
        s = np.linspace(0, 1, self.steps_per_segment)
        tb = blend_ratio
        traj = np.zeros((self.steps_per_segment, len(q_start)))
        for i, si in enumerate(s):
            if si < tb:
                traj[i] = q_start + dq * (si**2 / (2*tb))
            elif si <= 1 - tb:
                traj[i] = q_start + dq * (si - tb/2)
            else:
                traj[i] = q_end - dq * ((1-si)**2 / (2*tb))
        return traj

    def multi_segment(self, waypoints, method='quintic'):
        """多段插值，串联所有路径点"""
        segments = []
        for i in range(len(waypoints) - 1):
            if method == 'quintic':
                seg = self.quintic_segment(waypoints[i], waypoints[i+1])
            elif method == 'parabolic':
                seg = self.parabolic_segment(waypoints[i], waypoints[i+1])
            segments.append(seg)
        return np.vstack(segments)

    def total_joint_movement(self, waypoints):
        """计算路径总关节移动量"""
        total = 0
        for i in range(len(waypoints) - 1):
            total += np.sum(np.abs(waypoints[i+1] - waypoints[i]))
        return total

    # ── 碰撞检测 ──
    def check_collision(self, q, margin=0.05):
        """检查给定关节角是否与障碍物碰撞"""
        pin.forwardKinematics(self.kin.model, self.kin.data, q)
        pin.updateFramePlacements(self.kin.model, self.kin.data)

        link_positions = [
            self.kin.data.oMf[i].translation
            for i in range(len(self.kin.model.frames))
        ]

        for obs in self.obstacles.values():
            for pos in link_positions:
                if obs['type'] == 'box':
                    half = np.array(obs['size']) / 2 + margin
                    if np.all(np.abs(pos - obs['position']) < half):
                        return True
                elif obs['type'] == 'sphere':
                    if np.linalg.norm(pos - obs['position']) < obs['radius'] + margin:
                        return True
        return False

    # ── RRT 避障 ──
    def rrt(self, q_start, q_goal, max_iter=2000, step_size=0.1):
        """RRT 随机路径规划，返回无碰撞路径点列表"""
        lower = self.kin.model.lowerPositionLimit
        upper = self.kin.model.upperPositionLimit

        tree = [q_start]
        parent = {0: None}

        for it in range(max_iter):
            if it % 500 == 0 and it > 0:
                print(f"    RRT 进度: {it}/{max_iter} 树大小={len(tree)}")

            q_rand = q_goal if np.random.rand() < 0.1 else np.random.uniform(lower, upper)

            distances = [np.linalg.norm(q - q_rand) for q in tree]
            nearest_idx = np.argmin(distances)
            q_near = tree[nearest_idx]

            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            q_new = q_near + step_size * direction / dist

            if self.check_collision(q_new):
                continue

            tree.append(q_new)
            parent[len(tree) - 1] = nearest_idx

            if np.linalg.norm(q_new - q_goal) < step_size:
                path = [q_goal]
                idx = len(tree) - 1
                while parent[idx] is not None:
                    path.append(tree[idx])
                    idx = parent[idx]
                path.append(q_start)
                path.reverse()
                return path

        print("    RRT 未找到路径")
        # 返回最短路径（到目标最近的节点）
        best_idx = np.argmin([np.linalg.norm(tree[i] - q_goal) for i in range(len(tree))])
        if best_idx != 0:
            path = [q_goal]
            idx = best_idx
            while parent[idx] is not None:
                path.append(tree[idx])
                idx = parent[idx]
            path.append(q_start)
            path.reverse()
            print(f"    返回近似路径 (距目标 {np.linalg.norm(tree[best_idx] - q_goal):.4f})")
            return path
        return None

    # ── 关节索引 ──
    def _get_joint_idx(self, name):
        """通过关节名称获取在 q 向量中的索引"""
        for i in range(1, self.kin.model.njoints):
            if name in self.kin.model.names[i]:
                return self.kin.model.joints[i].idx_q
        return None

    def _get_gripper_q_index(self):
        idx = self._get_joint_idx("gripper")
        return idx if idx is not None else self.kin.model.nq - 1

    def _get_wrist_indices(self):
        """获取 wrist_flex 和 wrist_roll 在 q 中的索引"""
        flex = self._get_joint_idx("wrist_flex")
        roll = self._get_joint_idx("wrist_roll")
        return flex, roll

    # ── 腕部 IK（只调最后两个腕关节，保持手臂不变） ──
    def _solve_wrist_only(self, q_base, target_pos, target_rot,
                          max_iter=200, eps=1e-4, damping=0.1):
        """
        固定手臂关节 (q[0:3])，只解算腕部关节 (wrist_flex, wrist_roll)
        使末端到达 target_pos 且姿态为 target_rot。
        如果位置过远则返回 None。
        """
        flex_idx, roll_idx = self._get_wrist_indices()
        if flex_idx is None or roll_idx is None:
            return None

        q = q_base.copy()
        lower = self.kin.model.lowerPositionLimit
        upper = self.kin.model.upperPositionLimit
        target_se3 = pin.SE3(target_rot, np.array(target_pos))
        wrist_idxs = [flex_idx, roll_idx]

        for _ in range(max_iter):
            pin.forwardKinematics(self.kin.model, self.kin.data, q)
            pin.updateFramePlacements(self.kin.model, self.kin.data)

            current_se3 = self.kin.data.oMf[self.kin.ee_id]
            error = pin.log6(current_se3.inverse() * target_se3).vector

            if np.linalg.norm(error) < eps:
                return np.clip(q, lower, upper)

            J = pin.computeFrameJacobian(
                self.kin.model, self.kin.data, q, self.kin.ee_id,
                pin.ReferenceFrame.LOCAL
            )

            # 只取腕关节对应的雅可比列
            J_wrist = J[:, wrist_idxs]
            JwT = J_wrist.T
            dq_wrist = JwT @ np.linalg.inv(J_wrist @ JwT + damping**2 * np.eye(6)) @ error

            # 组装到完整 dq 中（非腕关节不变）
            dq_full = np.zeros(self.kin.model.nq)
            for idx, dq_val in zip(wrist_idxs, dq_wrist):
                dq_full[idx] = dq_val
            q = pin.integrate(self.kin.model, q, dq_full)
            q = np.clip(q, lower, upper)

        # 检查最终位置偏差
        pin.forwardKinematics(self.kin.model, self.kin.data, q)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        final_se3 = self.kin.data.oMf[self.kin.ee_id]
        err = np.linalg.norm(pin.log6(final_se3.inverse() * target_se3).vector)
        if err < 0.05:  # 5cm 以内可接受
            return q
        return None

    # ── 计算释放姿态（使 gripper 关节轴线竖直） ──
    def _compute_release_q(self, q_approach, target_pos):
        """
        从 approaching 姿态出发，使 gripper 关节轴线竖直。
        gripper 关节轴线在 gripper_frame 中沿 (0,-1,0) 方向，
        需要旋转 R_x(-π/2) 使其映射到世界 (0,0,1)（竖直）。
        """
        flex_idx, roll_idx = self._get_wrist_indices()
        if flex_idx is None or roll_idx is None:
            return None

        # R_x(-π/2)：使 gripper 关节轴线竖直
        theta = -np.pi / 2
        rot_vertical = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

        # 先用完整 IK 求解（从 approach 做初值，解收敛快）
        q_release, success = self.kin.inverse_kinematics(
            np.array(target_pos), rot_vertical, q_init=q_approach,
            max_iter=200, eps=1e-3
        )
        if success:
            return q_release

        # fallback: 只调腕部
        return self._solve_wrist_only(q_approach, target_pos, rot_vertical)

    # ── 主规划接口 ──
    def plan(self, target_pos, place_box_pos, place_box_size=None,
             method='quintic', n_rrt_trials=3):
        """
        完整规划：IK → RRT避障 → 插值
        在释放点只调节腕部使夹爪朝下（公垂线水平），
        保持手臂关节不变，确保释放位置准确。
        """
        if place_box_size is None:
            place_box_size = [0.08, 0.08, 0.04]

        # ── 1. 随机生成释放点（在盒子范围内） ──
        bx, by, bz = place_box_pos
        bw, bl, bh = place_box_size
        box_top = bz + bh / 2

        rx = bx + np.random.uniform(-bw/2, bw/2)
        ry = by + np.random.uniform(-bl/2, bl/2)
        rz = box_top + np.random.uniform(0.1, 0.3)

        print(f"\n释放点: ({rx:.3f}, {ry:.3f}, {rz:.3f})  "
              f"(盒子范围: x∈[{bx-bw/2:.3f},{bx+bw/2:.3f}], "
              f"y∈[{by-bl/2:.3f},{by+bl/2:.3f}], "
              f"z∈[{box_top+0.1:.3f},{box_top+0.3:.3f}])")

        # ── 2. 笛卡儿路径点（全部用单位姿态，IK 解手臂位置） ──
        cartesian_positions = [
            np.array([target_pos[0], target_pos[1], target_pos[2] + self.safe_height]),  # 0: 目标上方
            np.array([target_pos[0], target_pos[1], target_pos[2]]),                     # 1: 目标 (抓取)
            np.array([target_pos[0], target_pos[1], target_pos[2] + self.safe_height]),  # 2: 抬升
            np.array([rx, ry, rz + self.safe_height]),                                   # 3: 释放点上方 (approach)
            np.array([rx, ry, rz]),                                                      # 4: 释放点
        ]

        # ── 3. IK 求解（全部用单位姿态，后续再改腕部） ──
        q_waypoints = []
        q_current = pin.neutral(self.kin.model)
        for pos in cartesian_positions:
            q_sol, success = self.kin.inverse_kinematics(
                pos, np.eye(3), q_init=q_current
            )
            if not success:
                q_sol, _ = self.kin.inverse_kinematics(
                    pos, np.eye(3), q_init=None
                )
            lower = self.kin.model.lowerPositionLimit
            upper = self.kin.model.upperPositionLimit
            if np.any(q_sol < lower) or np.any(q_sol > upper):
                q_sol, success = self.kin.inverse_kinematics(
                    pos, np.eye(3), q_init=None
                )
                if not success:
                    print(f"IK 失败: {pos}")
                    return None
            q_waypoints.append(q_sol)
            q_current = q_sol

        # ── 4. 替换释放点：保持手臂不变，只调腕部使夹爪朝下 ──
        print("  调整腕部使夹爪朝下（公垂线水平）...")
        q_release = self._compute_release_q(q_waypoints[3], cartesian_positions[4])
        if q_release is not None:
            q_waypoints[4] = q_release
            # 验证腕部调整后的末端位置
            pin.forwardKinematics(self.kin.model, self.kin.data, q_release)
            pin.updateFramePlacements(self.kin.model, self.kin.data)
            actual_pos = self.kin.data.oMf[self.kin.ee_id].translation
            pos_err = np.linalg.norm(actual_pos - cartesian_positions[4])
            print(f"    释放点位置偏差: {pos_err:.4f}m")
        else:
            print("    腕部调整失败，使用原始 IK 结果")

        # ── 5. 分段 RRT 避障（多次尝试取最短路径） ──
        best_path = None
        best_cost = float('inf')

        for trial in range(n_rrt_trials):
            rrt_waypoints = [q_waypoints[0]]
            failed = False
            for i in range(len(q_waypoints) - 1):
                dist = np.linalg.norm(q_waypoints[i] - q_waypoints[i+1])
                if trial == 0:
                    print(f"  RRT 段 {i}: 关节空间距离 = {dist:.6f}")
                if dist < 0.2:
                    rrt_waypoints.append(q_waypoints[i+1])
                    continue
                path = self.rrt(q_waypoints[i], q_waypoints[i+1])
                if path is None:
                    failed = True
                    break
                rrt_waypoints.extend(path[1:])

            if failed:
                continue

            cost = self.total_joint_movement(rrt_waypoints)
            if cost < best_cost:
                best_cost = cost
                best_path = rrt_waypoints

        if best_path is None:
            print("RRT 规划失败")
            return None

        if n_rrt_trials > 1:
            print(f"  最优路径代价: {best_cost:.4f}")

        # ── 6. 生成平滑轨迹 ──
        trajectory = self.multi_segment(best_path, method)

        # ── 7. 在释放点处打开夹爪 ──
        gripper_idx = self._get_gripper_q_index()
        if gripper_idx is not None:
            # 在最后一段逐渐打开夹爪
            n_steps = len(trajectory)
            open_start = n_steps - self.steps_per_segment
            for i in range(open_start, n_steps):
                t = (i - open_start) / self.steps_per_segment
                trajectory[i, gripper_idx] = 1.74533 * t

        return {
            'waypoints': best_path,
            'trajectory': trajectory,
            'cost': best_cost,
            'release_point': (rx, ry, rz),
            'gripper_open': gripper_idx is not None,
        }