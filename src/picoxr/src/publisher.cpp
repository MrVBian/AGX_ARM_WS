#include <chrono>
#include <memory>
#include <iostream>
#include <functional>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <nlohmann/json.hpp>
#include <vector>
#include <iomanip>
#include <sstream>
#include <string>
#include <array>
#include "xr_msgs/msg/custom.hpp"
#include "xr_msgs/msg/head.hpp"
#include "xr_msgs/msg/controller.hpp"

#include "PXREARobotSDK.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/joint_state.hpp"  // 添加 JointState 头文件

#include <filesystem>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;
using namespace pinocchio; // 添加，便于直接使用 pinocchio 类型

struct IKResult {
  bool success;
  Eigen::VectorXd q;
  Eigen::VectorXd err;
  int iterations;
};

std::function<void(void* context, PXREAClientCallbackType type, int status, void* userData)> g_callback;

std::mutex g_callback_mutex;

void callbackForwarder(void* context, PXREAClientCallbackType type, int status, void* userData) {
  std::lock_guard<std::mutex> lock(g_callback_mutex);
  if (g_callback) {
    g_callback(context, type, status, userData);
  }
}

void print_json(const json& j, int indent=1) {
    // 根据缩进级别设置空格
    std::string indent_str(indent * 2, ' ');

    if (j.is_object()) {
        std::cout << indent_str << "{\n";
        for (auto it = j.begin(); it != j.end(); ++it) {
            std::cout << indent_str << "  \"" << it.key() << "\": ";
            print_json(it.value(), indent + 1);
            std::cout << ",\n";
        }
        // 删除最后一个多余的逗号
        if (!j.empty()) {
            std::cout << "\b\b" << std::endl;
        }
        std::cout << indent_str << "}";
    } else if (j.is_array()) {
        std::cout << indent_str << "[\n";
        for (const auto& item : j) {
            std::cout << indent_str << "  ";
            print_json(item, indent + 1);
            std::cout << ",\n";
        }
        if (!j.empty()) {
            std::cout << "\b\b" << std::endl;
        }
        std::cout << indent_str << "]";
    } else if (j.is_string()) {
        std::cout << "\"" << j.get<std::string>() << "\"";
    } else if (j.is_boolean()) {
        std::cout << (j.get<bool>() ? "true" : "false");
    } else if (j.is_number_integer()) {
        std::cout << j.get<int64_t>();
    } else if (j.is_number_unsigned()) {
        std::cout << j.get<uint64_t>();
    } else if (j.is_number_float()) {
        std::cout << j.get<double>();
    } else if (j.is_null()) {
        std::cout << "null";
    } else {
        std::cout << "unknown type";
    }
}


std::vector<float> stringToFloatVector(const std::string& input) {
    std::vector<float> result;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stof(token));
        } catch (const std::exception& e) {
            std::cerr << "转换错误: " << token << " -> " << e.what() << std::endl;
        }
    }
    return result;
}

///
/// 坐标系转换
///
// 新增：四元数/矩阵辅助函数，用于坐标系基变换（R' = M * R * M^T），保持物体朝向不变只是改表示基底
static std::array<double,9> quatToMat(double x, double y, double z, double w) {
  std::array<double,9> R;
  double xx = x*x, yy = y*y, zz = z*z;
  double xy = x*y, xz = x*z, xw = x*w, yz = y*z, yw = y*w, zw = z*w;
  R[0] = 1.0 - 2.0*(yy + zz);
  R[1] = 2.0*(xy - zw);
  R[2] = 2.0*(xz + yw);
  R[3] = 2.0*(xy + zw);
  R[4] = 1.0 - 2.0*(xx + zz);
  R[5] = 2.0*(yz - xw);
  R[6] = 2.0*(xz - yw);
  R[7] = 2.0*(yz + xw);
  R[8] = 1.0 - 2.0*(xx + yy);
  return R;
}

static std::array<double,9> changeBasis_M_R_Mt(const std::array<double,9>& R) {
  // M 定义：target_x = -original_z; target_y = original_x; target_z = original_y
  // 即 M = [[0,0,-1],[1,0,0],[0,1,0]]
  const double M[3][3] = {{0,0,-1},{1,0,0},{0,1,0}};
  std::array<double,9> tmp{};
  // tmp = M * R
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    double s = 0.0;
    for (int k=0;k<3;++k) s += M[i][k] * R[k*3 + j];
    tmp[i*3 + j] = s;
  }
  std::array<double,9> R2{};
  // R2 = tmp * M^T
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    double s = 0.0;
    for (int k=0;k<3;++k) s += tmp[i*3 + k] * M[j][k];
    R2[i*3 + j] = s;
  }
  return R2;
}

static std::array<double,4> matToQuat(const std::array<double,9>& R) {
  std::array<double,4> q; // x,y,z,w
  double trace = R[0] + R[4] + R[8];
  if (trace > 0.0) {
    double s = 0.5 / std::sqrt(trace + 1.0);
    q[3] = 0.25 / s;
    q[0] = (R[7] - R[5]) * s;
    q[1] = (R[2] - R[6]) * s;
    q[2] = (R[3] - R[1]) * s;
  } else {
    if (R[0] > R[4] && R[0] > R[8]) {
      double s = 2.0 * std::sqrt(1.0 + R[0] - R[4] - R[8]);
      q[3] = (R[7] - R[5]) / s;
      q[0] = 0.25 * s;
      q[1] = (R[1] + R[3]) / s;
      q[2] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
      double s = 2.0 * std::sqrt(1.0 + R[4] - R[0] - R[8]);
      q[3] = (R[2] - R[6]) / s;
      q[0] = (R[1] + R[3]) / s;
      q[1] = 0.25 * s;
      q[2] = (R[5] + R[7]) / s;
    } else {
      double s = 2.0 * std::sqrt(1.0 + R[8] - R[0] - R[4]);
      q[3] = (R[3] - R[1]) / s;
      q[0] = (R[2] + R[6]) / s;
      q[1] = (R[5] + R[7]) / s;
      q[2] = 0.25 * s;
    }
  }
  // 归一化以避免数值误差
  double norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  if (norm > 0.0) {
    q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
  }
  return q;
}

// 将 controller 四元数 (x,y,z,w) 从 controller 坐标系变换到 ps 所用坐标系
static std::array<double,4> convertControllerQuatToPoseQuat(float qx, float qy, float qz, float qw) {
  auto R = quatToMat(qx, qy, qz, qw);
  auto R2 = changeBasis_M_R_Mt(R);
  auto q_out = matToQuat(R2);
  return q_out; // x,y,z,w
}

///
///
///


class XRNode : public rclcpp::Node
{
public:
  XRNode() : Node("xr_publisher")
  {
    publisher_ = this->create_publisher<xr_msgs::msg::Custom>("xr_pose", 10);
    // 修改：将 pose_publisher_ 改为 joint_publisher_
    joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/control/move_j", 10);

    // 尝试加载 URDF 并构建 pinocchio 模型（仅一次）
    namespace fs = std::filesystem;
    const std::string urdf_filename = "/projects/xr/agx_arm_ws/src/piper/urdf/piper_description.urdf";
    if (!fs::exists(urdf_filename)) {
      RCLCPP_ERROR(this->get_logger(), "URDF not found: %s", urdf_filename.c_str());
      model_loaded_ = false;
    } else {
      try {
        model_ptr_ = std::make_unique<Model>();
        pinocchio::urdf::buildModel(urdf_filename, *model_ptr_);
        data_ptr_ = std::make_unique<Data>(*model_ptr_);
        model_loaded_ = true;
        RCLCPP_INFO(this->get_logger(), "Loaded URDF: %s  model.nq=%d nv=%d", urdf_filename.c_str(), model_ptr_->nq, model_ptr_->nv);
        
        // // 记录关节名称
        // for (int i = 0; i < model_ptr_->njoints; ++i) {
        //   RCLCPP_INFO(this->get_logger(), "Joint %d: %s", i, model_ptr_->names[i].c_str());
        // }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to build pinocchio model: %s", e.what());
        model_loaded_ = false;
      }
    }
    // 启动异步 IK 线程（仅在 model_loaded_ 时会执行求解）
    ik_thread_running_.store(true);
    ik_thread_ = std::thread([this]() { this->ikWorkerLoop(); });
  }

  ~XRNode() {
    // 停止线程
    ik_thread_running_.store(false);
    ik_cv_.notify_all();
    if (ik_thread_.joinable()) ik_thread_.join();
  }

  // 修改：solveIK 接受 Data 引用以复用 data_ptr_
  IKResult solveIK(const Model & model,
                  Data & data,
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

    // 使用外部传入的 data（避免每次分配）
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
    }

    res.q = q;
    res.err = err;
    return res;
  }

  // 获取活动关节的索引
  std::vector<int> getActiveJointIndices(const Model& model) {
    std::vector<int> active_joint_indices;
    for (int i = 0; i < model.njoints; ++i) {
      // 跳过固定关节（通常索引0是固定关节）
      if (model.joints[i].nq() > 0 && i > 0) {  // 通常索引0是root关节
        active_joint_indices.push_back(i);
      }
    }
    return active_joint_indices;
  }

  // 异步 IK 工作线程：等待最新目标，使用 model_ptr_ 和 data_ptr_ 求解，使用 warm start
  void ikWorkerLoop() {
    // 获取活动关节索引
    std::vector<int> active_joint_indices;
    std::vector<std::string> active_joint_names;
    if (model_loaded_ && model_ptr_) {
      active_joint_indices = getActiveJointIndices(*model_ptr_);
      for (int idx : active_joint_indices) {
        active_joint_names.push_back(model_ptr_->names[idx]);
      }
      // RCLCPP_INFO(this->get_logger(), "Active joints: %d joints", active_joint_indices.size());
      // for (size_t i = 0; i < active_joint_indices.size(); ++i) {
      //   RCLCPP_INFO(this->get_logger(), "  Joint %d: %s (nq=%d)", 
      //              active_joint_indices[i], active_joint_names[i].c_str(),
      //              model_ptr_->joints[active_joint_indices[i]].nq());
      // }
    }
    
    while (ik_thread_running_.load()) {
      std::unique_lock<std::mutex> lk(ik_mutex_);
      // 等待被通知或每 100ms 超时检查一次（避免永久阻塞）
      ik_cv_.wait_for(lk, std::chrono::milliseconds(100), [this]() {
        return !ik_thread_running_.load() || new_target_.load();
      });

      if (!ik_thread_running_.load()) break;
      if (!new_target_.load()) continue;

      // 拷贝目标并清标志（尽量缩短锁持有时间）
      SE3 target = latest_oMdes_;
      Eigen::VectorXd q_init = latest_q_init_;
      new_target_.store(false);
      lk.unlock();

      if (!model_loaded_ || !model_ptr_ || !data_ptr_) continue;

      // 使用 warm start：如果 last_solution_ 非空，使用它作为初始 q
      {
        std::lock_guard<std::mutex> lg(last_solution_mutex_);
        if (last_solution_.size() == model_ptr_->nq) {
          q_init = last_solution_;
        }
      }

      // 降低 IT_MAX 与松一点阈值以加速（可根据需要调整）
      IKResult ikres = solveIK(*model_ptr_, *data_ptr_, "link6", target, q_init, 2e-4, 300, 0.1, 1e-6);

      if (ikres.success) {
        // 存储最新解用于 warm start
        {
          std::lock_guard<std::mutex> lg(last_solution_mutex_);
          last_solution_ = ikres.q;
        }
        // RCLCPP_INFO(this->get_logger(), "Async IK converged iters=%d", ikres.iterations);
        
        // 发布关节状态消息
        auto joint_msg = std::make_unique<sensor_msgs::msg::JointState>();
        joint_msg->header.stamp = this->now();
        joint_msg->header.frame_id = "base_link";
        
        // 从 q 向量中提取活动关节的值
        for (size_t i = 0; i < active_joint_indices.size(); ++i) {
          int joint_idx = active_joint_indices[i];
          int q_idx = model_ptr_->idx_qs[joint_idx];
          
          // 获取关节名称
          joint_msg->name.push_back(active_joint_names[i]);
          
          // 获取关节位置
          joint_msg->position.push_back(ikres.q[q_idx]);
          
          // // 速度和力
          // joint_msg->velocity.push_back(0.0);
          // joint_msg->effort.push_back(0.0);
        }
        
        // 发布关节状态
        joint_publisher_->publish(std::move(joint_msg));
        
        // 创建详细的关节值字符串
        std::stringstream joint_values_ss;
        joint_values_ss << "Published joint states (" << active_joint_indices.size() << " joints): ";
        for (size_t i = 0; i < active_joint_indices.size(); ++i) {
          int joint_idx = active_joint_indices[i];
          int q_idx = model_ptr_->idx_qs[joint_idx];
          joint_values_ss << active_joint_names[i] << "=" << std::fixed << std::setprecision(4) << ikres.q[q_idx];
          if (i < active_joint_indices.size() - 1) {
            joint_values_ss << ", ";
          }
        }
        RCLCPP_INFO(this->get_logger(), "%s", joint_values_ss.str().c_str());
        
      } else {
        RCLCPP_DEBUG(this->get_logger(), "Async IK failed iters=%d", ikres.iterations);
      }
    }
  }

  // 修改回调部分：不直接求解 IK，只更新 latest target 并 notify 工作线程
  void OnPXREAClientCallback(void* context, PXREAClientCallbackType type,int status,void* userData)
  {
    (void)context;
    switch (type)
    {
      case PXREAServerConnect:
          std::cout <<"server connect"  << std::endl;
          break;
      case PXREAServerDisconnect:
          std::cout  <<"server disconnect"  << std::endl;
          break;
      case PXREADeviceFind:
          std::cout << "device find"<< (const char*)userData << std::endl;
          break;
      case PXREADeviceMissing:
          std::cout <<"device missing"<<(const char*)userData<<  std::endl;
          break;
      case PXREADeviceConnect:
          std::cout <<"device connect"<<(const char*)userData<<status<< std::endl;
          break;
      case PXREADeviceStateJson: {
          auto& dsj = *((PXREADevStateJson*)userData);
          try {
            auto json_obj = json::parse(dsj.stateJson);
            auto value_str = json_obj["value"].get<std::string>();
            auto value_obj = json::parse(value_str);

            auto custom_msg = xr_msgs::msg::Custom();
            custom_msg.timestamp_ns = value_obj["timeStampNs"].get<uint64_t>();
            custom_msg.input = value_obj["Input"].get<int>();

            // head
            auto head_msg = xr_msgs::msg::Head();
            if (value_obj.contains("Head")) {
              auto head_j = value_obj["Head"];
              std::vector<float> head_pose = stringToFloatVector(head_j["pose"].get<std::string>());
              if (head_pose.size() != 7) {
                std::cerr << "Parse failed: head pose data length != 7" << std::endl;
              }
              std::copy(head_pose.begin(), head_pose.end(), head_msg.pose.data());
              head_msg.status = head_j["status"].get<int>();
            } else {
              head_msg.status = -1;
            }
            custom_msg.head = head_msg;

            // controller
            if (value_obj.contains("Controller")) {
              for (auto& element : value_obj["Controller"].items()) {
                auto controller_msg = xr_msgs::msg::Controller();
                auto ctrl_j = element.value();

                controller_msg.axis_x = ctrl_j["axisX"].get<float>();
                controller_msg.axis_y = ctrl_j["axisY"].get<float>();
                controller_msg.axis_click = ctrl_j["axisClick"].get<bool>();
                controller_msg.gripper = ctrl_j["grip"].get<float>();
                controller_msg.trigger = ctrl_j["trigger"].get<float>();
                controller_msg.primary_button = ctrl_j["primaryButton"].get<bool>();
                controller_msg.secondary_button = ctrl_j["secondaryButton"].get<bool>();
                controller_msg.menu_button = ctrl_j["menuButton"].get<bool>();
                std::vector<float> ctrl_pose = stringToFloatVector(ctrl_j["pose"].get<std::string>());
                if (ctrl_pose.size() != 7) {
                  std::cerr << "Parse failed: ctrl pose data length != 7" << std::endl;
                }
                std::copy(ctrl_pose.begin(), ctrl_pose.end(), controller_msg.pose.data());
                controller_msg.status = 3;

                if (element.key() == "left") {
                  custom_msg.left_controller = controller_msg;
                } else {
                  custom_msg.right_controller = controller_msg;
                }
              }
            } else {
              auto left_controller_msg = xr_msgs::msg::Controller();
              auto right_controller_msg = xr_msgs::msg::Controller();
              left_controller_msg.status = -1;
              right_controller_msg.status = -1;
              custom_msg.left_controller = left_controller_msg;
              custom_msg.right_controller = right_controller_msg;
            }

            // hand

            // body

            // // XR
            publisher_->publish(custom_msg);

            if (custom_msg.left_controller.trigger == 1.0f) {
              geometry_msgs::msg::PoseStamped ps;
              ps.header.stamp = this->now();
              ps.header.frame_id = "Pico";
              // 位置坐标系映射：target_x = -original_z; target_y = original_x; target_z = original_y
              ps.pose.position.x = -custom_msg.left_controller.pose[2]; // 取反
              ps.pose.position.y = custom_msg.left_controller.pose[0];
              ps.pose.position.z = custom_msg.left_controller.pose[1];
              // 四元数转换（controller -> ps 坐标系）
              auto qconv = convertControllerQuatToPoseQuat(
                custom_msg.left_controller.pose[3],
                custom_msg.left_controller.pose[4],
                custom_msg.left_controller.pose[5],
                custom_msg.left_controller.pose[6]
              );
              ps.pose.orientation.x = static_cast<float>(qconv[0]);
              ps.pose.orientation.y = static_cast<float>(qconv[1]);
              ps.pose.orientation.z = static_cast<float>(qconv[2]);
              ps.pose.orientation.w = static_cast<float>(qconv[3]);
              
              RCLCPP_INFO(this->get_logger(), "Left controller pose: [%f, %f, %f, %f, %f, %f, %f]",
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
                ps.pose.orientation.w
              );

              // 求逆解
              if (model_loaded_ && model_ptr_) {
                try {
                  // 构建目标 SE3（注意 Eigen 四元数构造顺序：w,x,y,z）
                  Eigen::Quaterniond quat_target(
                    ps.pose.orientation.w,
                    ps.pose.orientation.x,
                    ps.pose.orientation.y,
                    ps.pose.orientation.z
                  );
                  quat_target.normalize();
                  SE3 oMdes(quat_target.toRotationMatrix(),
                            Eigen::Vector3d(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z));
                  // 将目标交给异步 IK 线程处理（避免在 90Hz 回调中阻塞）
                  {
                    std::lock_guard<std::mutex> lk(ik_mutex_);
                    latest_oMdes_ = oMdes;
                    // 使用 neutral 作为初始 guess，工作线程会用 last_solution_ 进行 warm start（若可用）
                    latest_q_init_ = neutral(*model_ptr_);
                    new_target_.store(true);
                  }
                  ik_cv_.notify_one();
                } catch (const std::exception & e) {
                  RCLCPP_ERROR(this->get_logger(), "IK exception: %s", e.what());
                }
              } else {
                RCLCPP_WARN(this->get_logger(), "Pinocchio model not loaded, cannot run IK.");
              }
            }

          } catch (const std::exception& e) {
            std::cerr << "Parse failed: " << e.what() << std::endl;
          }
          break;
      }
      case PXREADeviceCustomMessage:
          std::cout << "device custom message" << std::endl;
          break;
      case PXREAFullMask:
          std::cout << "full mask" << std::endl;
          break;
    }
  }

private:
  rclcpp::Publisher<xr_msgs::msg::Custom>::SharedPtr publisher_;
  // 修改：将 pose_publisher_ 改为 joint_publisher_
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher_;

  // pinocchio 模型（可选）
  std::unique_ptr<Model> model_ptr_;
  std::unique_ptr<Data> data_ptr_;
  bool model_loaded_ = false;

  // IK 异步线程与目标缓存
  std::thread ik_thread_;
  std::atomic<bool> ik_thread_running_{false};
  std::condition_variable ik_cv_;
  std::mutex ik_mutex_;
  std::atomic<bool> new_target_{false};
  SE3 latest_oMdes_;               // 受 ik_mutex_ 保护
  Eigen::VectorXd latest_q_init_;  // 受 ik_mutex_ 保护

  // last solution 用作 warm start
  Eigen::VectorXd last_solution_;
  std::mutex last_solution_mutex_;

  // 辅助：将 Eigen 向量转为字符串用于日志（短小实现）
  std::string eigenVectorToString(const Eigen::VectorXd & v) {
    std::ostringstream oss;
    for (int i=0;i<v.size();++i) {
      if (i) oss << ", ";
      oss << std::fixed << std::setprecision(6) << v[i];
    }
    return oss.str();
  }
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto xrNode = std::make_shared<XRNode>();
  g_callback = [&xrNode] (void* context, PXREAClientCallbackType type,int status,void* userData) {
    xrNode->OnPXREAClientCallback(context, type, status, userData);
  };
  PXREAInit(NULL, callbackForwarder, PXREAFullMask);

  rclcpp::spin(xrNode);

  PXREADeinit();

  rclcpp::shutdown();
  return 0;
}