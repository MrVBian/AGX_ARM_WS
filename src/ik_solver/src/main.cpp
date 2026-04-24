/**
 * @file ik_example.cpp
 * @brief 使用Pinocchio库实现逆运动学（IK）求解示例
 * 
 * 该程序通过迭代雅可比伪逆方法，计算机器人末端执行器到达目标位姿所需的关节角度。
 * 目标位姿可通过命令行参数指定，否则使用默认值。
 * 注：本示例中机器人末端执行器为"link6"。
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"

using namespace pinocchio;

struct IKResult {
  bool success;
  Eigen::VectorXd q;
  Eigen::VectorXd err;
  int iterations;
};

// 封装的 IK 求解函数：传入 model、末端名称、目标位姿 oMdes、初始 q（可用 neutral(model)）
// 返回 IKResult（包含是否收敛、最终 q、最终误差、迭代次数）
IKResult solveIK(const Model & model,
                 const std::string & ee_name,
                 const SE3 & oMdes,
                 const Eigen::VectorXd & q_init,
                 double eps = 1e-4,
                 int IT_MAX = 1000,
                 double DT = 1e-1,
                 double damp = 1e-6)
{
  IKResult res;
  res.success = false;
  res.iterations = 0;

  // 查找末端帧
  FrameIndex ee_frame = FrameIndex(-1);
  for (FrameIndex i = 0; i < static_cast<FrameIndex>(model.nframes); ++i) {
    if (model.frames[i].name == ee_name) { ee_frame = i; break; }
  }
  if (ee_frame == FrameIndex(-1)) {
    res.err = Eigen::VectorXd::Zero(6);
    return res;
  }
  const JointIndex JOINT_ID = model.frames[ee_frame].parent;

  Data data(model);
  Eigen::VectorXd q = q_init;
  typedef Eigen::Matrix<double,6,1> Vector6d;
  Vector6d err = Vector6d::Zero();

  Data::Matrix6x J(6, model.nv); J.setZero();

  for (int iter = 0; ; ++iter) {
    forwardKinematics(model, data, q);
    const SE3 iMd = data.oMi[JOINT_ID].actInv(oMdes);
    err = log6(iMd).toVector();
    const double err_norm = err.norm();

    if (err_norm < eps) {
      res.success = true;
      res.iterations = iter;
      break;
    }
    if (iter >= IT_MAX) {
      res.success = false;
      res.iterations = iter;
      break;
    }

    computeJointJacobian(model, data, q, JOINT_ID, J);

    Data::Matrix6 Jlog; Jlog.setZero();
    Jlog6(iMd.inverse(), Jlog);
    J = -Jlog * J;

    Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += damp;
    Eigen::VectorXd v(model.nv);
    v.noalias() = -J.transpose() * JJt.ldlt().solve(err);

    q = integrate(model, q, v * DT);

    if ((iter % 10) == 0) {
      std::cout << "iter " << iter << " err_norm=" << err_norm << std::endl;
    }
  }

  res.q = q;
  res.err = err;
  return res;
}

int main(int argc, char ** argv)
{
  namespace fs = std::filesystem;
  using namespace pinocchio;

  // 1. 检查URDF文件路径并加载模型
  const std::string urdf_filename = "/projects/xr/agx_arm_ws/src/piper/urdf/piper_description.urdf";
  if (!fs::exists(urdf_filename)) {
    std::cerr << "URDF not found: " << urdf_filename << std::endl;
    return 1;
  }
  std::cout << "Using URDF: " << urdf_filename << std::endl;

  try {
    // 2. 从URDF文件构建机器人模型
    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);
    std::cout << "Model name: " << model.name << " nq=" << model.nq << " nv=" << model.nv << std::endl;

    // 设置期望位姿（示例：可从命令行读取）
    double tx=1.0, ty=0.0, tz=1.0;
    double qx=0.0, qy=0.0, qz=0.0, qw=1.0;
    if (argc >= 8) {
      tx = std::stod(argv[1]); ty = std::stod(argv[2]); tz = std::stod(argv[3]);
      qx = std::stod(argv[4]); qy = std::stod(argv[5]); qz = std::stod(argv[6]); qw = std::stod(argv[7]);
    }
    Eigen::Quaterniond quat(qw, qx, qy, qz); quat.normalize();
    SE3 oMdes(quat.toRotationMatrix(), Eigen::Vector3d(tx, ty, tz));
    const std::string ee_name = "link6";

    // 调用封装的 IK 求解函数（使用 neutral 作为初始 q）
    Eigen::VectorXd q_init = neutral(model);
    IKResult result = solveIK(model, ee_name, oMdes, q_init);

    if (result.success) {
      std::cout << "IK Converged in " << result.iterations << " iterations.\n";
      std::cout << "result q: " << result.q.transpose() << std::endl;
      std::cout << "final error: " << result.err.transpose() << std::endl;
    } else {
      std::cout << "IK failed (iterations=" << result.iterations << ").\n";
      std::cout << "last q: " << result.q.transpose() << std::endl;
      std::cout << "last error: " << result.err.transpose() << std::endl;
    }

  } catch (const std::exception & e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}