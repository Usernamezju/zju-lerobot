# kinematics.py
import pinocchio as pin
import numpy as np

class Kinematics:
    def __init__(self, urdf_path, mesh_dir):
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs=[mesh_dir]
        )
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId("gripper_frame_link")

    def forward_kinematics(self, q):
        """
        正运动学：给定关节角，返回末端位置和姿态
        输入: q -> 关节角数组 (nq,)
        输出: position (3,), rotation (3,3)
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        position = self.data.oMf[self.ee_id].translation
        rotation = self.data.oMf[self.ee_id].rotation
        return position.copy(), rotation.copy()

    def inverse_kinematics(self, target_pos, target_rot,
                           q_init=None, max_iter=1000, eps=1e-4, damping=0.1):
        """
        逆运动学：给定目标位姿，返回关节角
        使用阻尼最小二乘法防止关节角跑飞
        参数:
            damping: 阻尼系数，越大越稳定但收敛越慢
        返回: q (nq,), success (bool)
        """
        # 初始猜测
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = q_init.copy()

        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit

        # 目标位姿
        target_SE3 = pin.SE3(target_rot, target_pos)

        for i in range(max_iter):
            # 正运动学更新
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # 计算误差 Δx（在SE3流形上）
            current_SE3 = self.data.oMf[self.ee_id]
            error = pin.log6(current_SE3.inverse() * target_SE3).vector

            # 判断收敛
            if np.linalg.norm(error) < eps:
                # 收敛后将关节角钳位到限位内
                q = np.clip(q, lower, upper)
                return q, True

            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_id,
                pin.ReferenceFrame.LOCAL
            )

            # 阻尼最小二乘法: J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹
            # 相比纯伪逆，阻尼项防止了奇异点附近的大幅关节角跳变
            Jt = J.T
            dq = Jt @ np.linalg.inv(J @ Jt + damping**2 * np.eye(6)) @ error
            q = pin.integrate(self.model, q, dq)

            # 每 50 步矫正一次关节角限位，防止长期累积漂移
            if i > 0 and i % 50 == 0:
                if np.any(q < lower) or np.any(q > upper):
                    q = np.clip(q, lower, upper)

        print("IK 未收敛")
        # 最终钳位到关节限位
        q = np.clip(q, lower, upper)
        return q, False


if __name__ == "__main__":
    urdf_path = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    mesh_dir = "/home/fjp/lerobot/SO-ARM100/Simulation/SO101/"
    
