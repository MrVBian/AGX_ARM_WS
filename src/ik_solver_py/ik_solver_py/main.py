#!/usr/bin/env python3
"""
Pinocchio逆运动学求解器ROS2节点封装
监听目标位姿话题，求解逆运动学，然后发布关节状态
"""

import numpy as np
import pinocchio
from numpy.linalg import norm, solve
import warnings
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class InverseKinematicsSolver:
    """逆运动学求解器类"""
    
    def __init__(self, urdf_path=None, joint_id=6, damping=1e-12):
        """
        初始化逆运动学求解器
        
        参数:
            urdf_path: str, URDF文件路径。如果为None，则使用示例模型
            joint_id: int, 目标关节ID（默认6，对应末端执行器）
            damping: float, 阻尼系数，用于数值稳定性
        """
        if urdf_path is not None:
            # 从URDF文件加载模型
            self.model = pinocchio.buildModelFromUrdf(urdf_path)
        else:
            # 使用Pinocchio示例模型
            self.model = pinocchio.buildSampleModelManipulator()
        
        self.joint_id = joint_id
        self.damping = damping
        
        # 初始化数据
        self.data = self.model.createData()
        
        # 获取关节名称（排除固定关节）
        self.joint_names = self._get_joint_names()
    
    def _get_joint_names(self):
        """获取可动关节的名称列表"""
        joint_names = []
        for name in self.model.names[1:]:  # 跳过基座关节"universe"
            joint_id = self.model.getJointId(name)
            joint = self.model.joints[joint_id]
            if joint.nq > 0:  # 只考虑有自由度的关节，nq是属性不是方法
                joint_names.append(name)
        return joint_names
    
    def solve_ik(self, target_pose, initial_q=None, eps=1e-4, max_iter=1000, dt=1e-1, verbose=False):
        """
        求解逆运动学
        
        参数:
            target_pose: pinocchio.SE3, 目标位姿
            initial_q: np.ndarray, 初始关节角度。如果为None，则使用中性位置
            eps: float, 收敛阈值
            max_iter: int, 最大迭代次数
            dt: float, 步长
            verbose: bool, 是否打印迭代信息
            
        返回:
            tuple: (q, success, error_norm, iterations)
                - q: np.ndarray, 求解得到的关节角度
                - success: bool, 是否成功收敛
                - error_norm: float, 最终误差范数
                - iterations: int, 实际迭代次数
        """
        # 初始化关节角度
        if initial_q is None:
            q = pinocchio.neutral(self.model)
        else:
            q = initial_q.copy()
        
        # 重置数据
        self.data = self.model.createData()
        
        # 迭代求解
        for i in range(max_iter):
            # 前向运动学
            pinocchio.forwardKinematics(self.model, self.data, q)
            
            # 计算当前位姿与目标位姿的误差
            iMd = self.data.oMi[self.joint_id].actInv(target_pose)
            err = pinocchio.log(iMd).vector  # 在关节坐标系中
            
            # 计算误差范数
            error_norm = norm(err)
            
            # 检查收敛
            if error_norm < eps:
                if verbose:
                    print(f"Convergence achieved in {i} iterations!")
                return q, True, error_norm, i
            
            # 计算雅可比矩阵
            J = pinocchio.computeJointJacobian(self.model, self.data, q, self.joint_id)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            
            # 阻尼最小二乘法求解关节速度
            v = -J.T.dot(solve(J.dot(J.T) + self.damping * np.eye(6), err))
            
            # 积分得到新的关节角度
            q = pinocchio.integrate(self.model, q, v * dt)
            
            # 打印迭代信息
            if verbose and (i % 10 == 0 or i == 0):
                print(f"Iteration {i}: error = {error_norm:.6e}")
        
        # 如果达到最大迭代次数仍未收敛
        warnings.warn(
            f"\nWarning: Maximum iterations ({max_iter}) reached without convergence. "
            f"Final error: {error_norm:.6e}"
        )
        
        return q, False, error_norm, max_iter


class InverseKinematicsNode(Node):
    """逆运动学ROS2节点"""
    
    def __init__(self, node_name='inverse_kinematics_node'):
        super().__init__(node_name)
        
        # 声明参数
        self.declare_parameter('urdf_path', '/projects/xr/agx_arm_ws/src/piper/urdf/piper_description.urdf')
        self.declare_parameter('target_joint_id', 6)
        self.declare_parameter('damping', 1e-12)
        self.declare_parameter('eps', 1e-4)
        self.declare_parameter('max_iter', 1000)
        self.declare_parameter('dt', 1e-1)
        
        # 获取参数
        urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        joint_id = self.get_parameter('target_joint_id').get_parameter_value().integer_value
        damping = self.get_parameter('damping').get_parameter_value().double_value
        
        if not urdf_path:
            self.get_logger().error('URDF path not provided. Please set the urdf_path parameter.')
            raise ValueError('URDF path not provided.')
        
        # 初始化逆运动学求解器
        self.ik_solver = InverseKinematicsSolver(
            urdf_path=urdf_path,
            joint_id=joint_id,
            damping=damping
        )
        
        # 获取关节名称
        self.joint_names = self.ik_solver.joint_names
        
        # 初始化当前关节角度
        self.current_q = pinocchio.neutral(self.ik_solver.model)
        
        # 创建订阅者
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/xr/move_p',
            self.pose_callback,
            10
        )
        
        # sim
        self.joint_state_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # real
        # self.joint_state_publisher = self.create_publisher(
        #     JointState,
        #     '/control/move_j',
        #     10
        # )
        
        self.get_logger().info(f'逆运动学节点已启动，监听 /xr/move_p 话题')
        self.get_logger().info(f'关节数量: {len(self.joint_names)}')
        self.get_logger().info(f'关节名称: {self.joint_names}')
    
    def pose_callback(self, msg):
        """处理接收到的目标位姿消息"""
        self.get_logger().info('接收到新的目标位姿，开始逆运动学求解...')
        
        # 提取目标位置
        target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # 提取目标姿态（四元数）
        quaternion = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
        
        # 将四元数转换为旋转矩阵
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat([quaternion[1], quaternion[2], 
                                      quaternion[3], quaternion[0]]).as_matrix()
        
        # 创建目标位姿
        target_pose = pinocchio.SE3(rotation, target_position)
        
        # 获取求解参数
        eps = self.get_parameter('eps').get_parameter_value().double_value
        max_iter = self.get_parameter('max_iter').get_parameter_value().integer_value
        dt = self.get_parameter('dt').get_parameter_value().double_value
        
        # 求解逆运动学
        q, success, error_norm, iterations = self.ik_solver.solve_ik(
            target_pose=target_pose,
            initial_q=self.current_q,
            eps=eps,
            max_iter=max_iter,
            dt=dt,
            verbose=False
        )
        
        if success:
            self.get_logger().info(f'逆运动学求解成功! 迭代次数: {iterations}, 最终误差: {error_norm:.6e}')
            
            # 更新当前关节角度
            self.current_q = q.copy()
            
            # 发布关节状态
            self.publish_joint_state(q)
        else:
            self.get_logger().warn(f'逆运动学求解失败! 迭代次数: {iterations}, 最终误差: {error_norm:.6e}')
    
    def publish_joint_state(self, joint_positions):
        """发布关节状态消息"""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = 'base_link'
        
        # 设置关节名称
        joint_state_msg.name = self.joint_names
        
        # 获取可动关节的索引
        movable_joint_indices = []
        positions = []
        
        # 遍历所有关节，只提取可动关节的角度
        q_idx = 0
        for i, joint in enumerate(self.ik_solver.model.joints):
            if i == 0:  # 跳过第一个关节（universe）
                continue
                
            if joint.nq > 0:  # 可动关节
                movable_joint_indices.append(i)
                # 提取当前关节对应的角度
                for j in range(joint.nq):
                    positions.append(joint_positions[q_idx + j])
                q_idx += joint.nq
        
        joint_state_msg.position = positions
        
        # 设置速度和力为零
        joint_state_msg.velocity = [0.0] * len(positions)
        joint_state_msg.effort = [0.0] * len(positions)
        
        # 发布消息
        self.joint_state_publisher.publish(joint_state_msg)
        self.get_logger().info(f'关节状态已发布，关节数量: {len(positions)}')
        
        # 打印关节角度
        joint_angles_str = ', '.join([f'{angle:.4f}' for angle in positions])
        self.get_logger().info(f'关节角度: [{joint_angles_str}]')


def solve_inverse_kinematics(target_position, target_rotation=None, 
                            urdf_path=None, joint_id=6, initial_q=None,
                            eps=1e-4, max_iter=1000, dt=1e-1, 
                            damping=1e-12, verbose=False):
    """
    逆运动学求解函数（简化接口）
    
    参数:
        target_position: list/np.ndarray, 目标位置 [x, y, z]
        target_rotation: list/np.ndarray/None, 目标旋转矩阵(3x3)或四元数(w,x,y,z)
                         如果为None，则使用单位旋转矩阵
        urdf_path: str, URDF文件路径
        joint_id: int, 目标关节ID
        initial_q: list/np.ndarray, 初始关节角度
        eps: float, 收敛阈值
        max_iter: int, 最大迭代次数
        dt: float, 步长
        damping: float, 阻尼系数
        verbose: bool, 是否打印详细信息
        
    返回:
        tuple: (joint_angles, success, error_norm, iterations)
    """
    # 创建目标位姿
    if target_rotation is None:
        # 默认使用单位旋转矩阵
        rotation = np.eye(3)
    else:
        target_rotation = np.array(target_rotation)
        
        if target_rotation.shape == (3, 3):
            # 3x3旋转矩阵
            rotation = target_rotation
        elif target_rotation.size == 4:
            # 四元数 [w, x, y, z]
            from scipy.spatial.transform import Rotation
            quat = target_rotation.flatten()
            rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        elif target_rotation.size == 9:
            # 展平的3x3旋转矩阵
            rotation = target_rotation.reshape(3, 3)
        else:
            raise ValueError(
                f"target_rotation should be either 3x3 rotation matrix or quaternion (4 elements). "
                f"Got shape: {target_rotation.shape}"
            )
    
    # 创建SE3位姿
    target_pose = pinocchio.SE3(rotation, np.array(target_position))
    
    # 创建求解器
    ik_solver = InverseKinematicsSolver(urdf_path=urdf_path, 
                                       joint_id=joint_id, 
                                       damping=damping)
    
    # 求解逆运动学
    return ik_solver.solve_ik(target_pose, initial_q=initial_q, 
                             eps=eps, max_iter=max_iter, dt=dt, 
                             verbose=verbose)


def main(args=None):
    """主函数，启动ROS2节点"""
    rclpy.init(args=args)
    
    try:
        ik_node = InverseKinematicsNode()
        rclpy.spin(ik_node)
    except ValueError as e:
        ik_node.get_logger().error(str(e))
    except KeyboardInterrupt:
        ik_node.get_logger().info('节点被用户中断')
    finally:
        if 'ik_node' in locals():
            ik_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()