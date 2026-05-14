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
#include "std_srvs/srv/empty.hpp"

#include <filesystem>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

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

class XRNode : public rclcpp::Node
{
public:
  XRNode() : Node("xr_publisher")
  {
    publisher_ = this->create_publisher<xr_msgs::msg::Custom>("xr_pose", 10);
    
    // sim
    l_joint_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/xr/move_p", 10);
    
    
    // real
    // l_joint_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/control/move_p", 10);
    r_joint_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/right/control/move_p", 10);
    g_joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/control/joint_states", 10);
    // l_joint_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    l_real_pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/feedback/tcp_pose", 10, std::bind(&XRNode::lPoseCallback, this, std::placeholders::_1));
    void lPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    // 创建紧急停止服务客户端
    emergency_stop_client_ = this->create_client<std_srvs::srv::Empty>("/emergency_stop");

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

        // 获取关节的上下限
        int nq = model_ptr_->nq;  // 配置空间维度
        // 初始化 Eigen 向量
        joint_position_lower_limits_ = Eigen::VectorXd(nq);
        joint_position_upper_limits_ = Eigen::VectorXd(nq);
        // 从模型中获取限制
        // 注意：Pinocchio 模型将限制存储在模型中
        joint_position_lower_limits_ = model_ptr_->lowerPositionLimit;
        joint_position_upper_limits_ = model_ptr_->upperPositionLimit;
        // 输出关节上下限
        RCLCPP_INFO(this->get_logger(), "关节位置限制:");
        for (int i = 0; i < nq; ++i) {
          RCLCPP_INFO(this->get_logger(), "  关节[%d]: 下限=%f, 上限=%f", 
            i, 
            joint_position_lower_limits_[i], 
            joint_position_upper_limits_[i]);
        }
        
        // 记录关节名称
        for (int i = 0; i < model_ptr_->njoints; ++i) {
          RCLCPP_INFO(this->get_logger(), "Joint %d: %s", i, model_ptr_->names[i].c_str());
        }
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


    /**
   * @brief 使用Pinocchio库求解逆运动学（Inverse Kinematics）
   * 
   * @param model 机器人模型，包含关节、连杆、惯性参数等信息
   * @param data 机器人数据，用于存储计算过程中的中间结果
   * @param ee_name 末端执行器（end effector）的名称
   * @param oMdes 期望的末端位姿（相对于世界坐标系）
   * @param q_init 初始关节位置向量，作为迭代优化的起点
   * @param eps 收敛精度阈值，当末端误差小于此值时认为求解成功（默认1e-2）
   * @param IT_MAX 最大迭代次数，防止无限循环（默认1000）
   * @param DT 迭代步长，控制每次迭代关节位置的更新幅度（默认0.1）
   * @param damp 阻尼系数，用于求解雅可比矩阵伪逆时的正则化，避免奇异位置（默认1e-6）
   * 
   * @return IKResult 逆运动学求解结果，包含：
   *         - success: 是否成功求解
   *         - iterations: 实际迭代次数
   *         - q: 求解得到的关节位置
   *         - err: 最终的末端位置/姿态误差向量
   */
  IKResult solveIK(const Model & model,
                  Data & data,
                  const std::string & ee_name,
                  const SE3 & oMdes,
                  const Eigen::VectorXd & q_init,
                  double eps = 1e-1,
                  int IT_MAX = 1500,
                  double DT = 1e-1,
                  double damp = 1e-4)
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
    

    Data::Matrix6x J(6, model.nv); 
    J.setZero();

    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d err;
    Eigen::VectorXd v(model.nv);

    for (int iter = 0; ; ++iter) {
      forwardKinematics(model, data, q);
      const SE3 iMd = data.oMi[JOINT_ID].actInv(oMdes);
      err = log6(iMd).toVector();
      const double err_norm = err.norm();

      bool is_within_limits = true;
      int nq = model_ptr_->nq;  // 配置空间维度

      for (int i = 0; i < nq; ++i) {
        if (q[i] < joint_position_lower_limits_[i] || q[i] > joint_position_upper_limits_[i]) {
          is_within_limits = false;
          break;
        }
      }

      if (err_norm < eps && is_within_limits) {
        res.success = true;
        res.iterations = iter;
        break;
      }
      if (iter >= IT_MAX || !is_within_limits) {
        res.success = false;
        res.iterations = iter;
        break;
      }

      computeJointJacobian(model, data, q, JOINT_ID, J);

      Data::Matrix6 Jlog;
      Jlog.setZero();
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
      IKResult ikres = solveIK(*model_ptr_, *data_ptr_, "link6", target, q_init);

      if (ikres.success) {
        // 存储最新解用于 warm start
        {
          std::lock_guard<std::mutex> lg(last_solution_mutex_);
          last_solution_ = ikres.q;
        }
        // RCLCPP_INFO(this->get_logger(), "Async IK converged iters=%d", ikres.iterations);
        
        // // 发布关节状态消息
        // auto joint_msg = std::make_unique<sensor_msgs::msg::JointState>();
        // joint_msg->header.stamp = this->now();
        // joint_msg->header.frame_id = "base_link";
        
        // // 从 q 向量中提取活动关节的值
        // for (size_t i = 0; i < active_joint_indices.size(); ++i) {
        //   int joint_idx = active_joint_indices[i];
        //   int q_idx = model_ptr_->idx_qs[joint_idx];
          
        //   // 获取关节名称
        //   joint_msg->name.push_back(active_joint_names[i]);
          
        //   // 获取关节位置
        //   joint_msg->position.push_back(ikres.q[q_idx]);
          
        //   // // 速度和力
        //   // joint_msg->velocity.push_back(0.0);
        //   // joint_msg->effort.push_back(0.0);
        // }
        
        // // 发布关节状态
        // l_joint_publisher_->publish(std::move(joint_msg));
        

        const auto &t = target.translation();
        const auto &R = target.rotation();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "IK Target - Position: ["
          << t.x() << ", " << t.y() << ", " << t.z() << "], ";
        // 可以打印旋转矩阵（简单直接）
        ss << "Rotation matrix: ["
          << R(0,0) << ", " << R(0,1) << ", " << R(0,2) << "; "
          << R(1,0) << ", " << R(1,1) << ", " << R(1,2) << "; "
          << R(2,0) << ", " << R(2,1) << ", " << R(2,2) << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

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



  void lPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (l_real_has_new_pose_ == false){
      l_real_pose.pose.position.x = msg->pose.position.x;
      l_real_pose.pose.position.y = msg->pose.position.y;
      l_real_pose.pose.position.z = msg->pose.position.z;
      l_real_pose.pose.orientation.x = msg->pose.orientation.x;
      l_real_pose.pose.orientation.y = msg->pose.orientation.y;
      l_real_pose.pose.orientation.z = msg->pose.orientation.z;
      l_real_pose.pose.orientation.w = msg->pose.orientation.w;
      l_real_has_new_pose_ = true;

      // 打印接收到的信息
      RCLCPP_INFO(this->get_logger(), "\033[1;33mTCP位姿 topic: %s\n- 位置: [x: %.3f, y: %.3f, z: %.3f]\n姿态: [qx: %.3f, qy: %.3f, qz: %.3f, qw: %.3f]\033[0m", 
                  l_real_pose_subscriber_->get_topic_name(),
                  l_real_pose.pose.position.x, l_real_pose.pose.position.y, l_real_pose.pose.position.z,
                  l_real_pose.pose.orientation.x, l_real_pose.pose.orientation.y, 
                  l_real_pose.pose.orientation.z, l_real_pose.pose.orientation.w
      );
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

            // // XR
            publisher_->publish(custom_msg);

            if (custom_msg.left_controller.trigger == 1.0f) {
              geometry_msgs::msg::PoseStamped ps;
              ps.header.stamp = this->now();
              ps.header.frame_id = "Pico";

              ps.pose.position.x = custom_msg.left_controller.pose[0];
              ps.pose.position.y = custom_msg.left_controller.pose[1];
              ps.pose.position.z = custom_msg.left_controller.pose[2];
              ps.pose.orientation.x = custom_msg.left_controller.pose[3];
              ps.pose.orientation.y = custom_msg.left_controller.pose[4];
              ps.pose.orientation.z = custom_msg.left_controller.pose[5];
              ps.pose.orientation.w = custom_msg.left_controller.pose[6];

              if (l_ctl_init == false) {
                // 更新真机姿态
                l_real_has_new_pose_ = false;
                l_ctl_init_pose.header.stamp = ps.header.stamp;
                l_ctl_init_pose.header.frame_id = ps.header.frame_id;
                l_ctl_init_pose.pose.position.x = ps.pose.position.x;
                l_ctl_init_pose.pose.position.y = ps.pose.position.y;
                l_ctl_init_pose.pose.position.z = ps.pose.position.z;
                l_ctl_init_pose.pose.orientation.x = ps.pose.orientation.x;
                l_ctl_init_pose.pose.orientation.y = ps.pose.orientation.y;
                l_ctl_init_pose.pose.orientation.z = ps.pose.orientation.z;
                l_ctl_init_pose.pose.orientation.w = ps.pose.orientation.w;
                l_ctl_init = true;

                RCLCPP_INFO(this->get_logger(), 
                  "\033[1;33m[LEFT CTRL INIT] Pose: [x:%f, y:%f, z:%f, qx:%f, qy:%f, qz:%f, qw:%f]\033[0m",
                  ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                  ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
                  ps.pose.orientation.w
                );
              }
              else if(l_ctl_init == true && l_real_has_new_pose_ == true) {











  // 把控制器(右手系x右，y上，z后)里的移动和旋转，"翻译"成真实世界(右手系x前，y左，z上)里的动作
  // 然后叠加到当前的物体位置上，最后发布这个结果result
  
  // 1. 计算控制器相对于初始位置的位移
  double dx_ctl = ps.pose.position.x - l_ctl_init_pose.pose.position.x;
  double dy_ctl = ps.pose.position.y - l_ctl_init_pose.pose.position.y;
  double dz_ctl = ps.pose.position.z - l_ctl_init_pose.pose.position.z;
  
  // 2. 将控制器位移转换到真实世界坐标系
  // 控制器x+ -> 真实世界y-
  // 控制器y+ -> 真实世界z+
  // 控制器z+ -> 真实世界x-
  double dx_real = -dz_ctl;  // 控制器z+ -> 真实世界x-
  double dy_real = -dx_ctl;  // 控制器x+ -> 真实世界y-
  double dz_real = dy_ctl;   // 控制器y+ -> 真实世界z+
  
  // 3. 计算旋转变换
  // 获取控制器初始和当前的四元数
  tf2::Quaternion q_ctl_init(
    l_ctl_init_pose.pose.orientation.x,
    l_ctl_init_pose.pose.orientation.y,
    l_ctl_init_pose.pose.orientation.z,
    l_ctl_init_pose.pose.orientation.w
  );
  
  tf2::Quaternion q_ctl_current(
    ps.pose.orientation.x,
    ps.pose.orientation.y,
    ps.pose.orientation.z,
    ps.pose.orientation.w
  );
  















// 计算相对旋转：从初始到当前的旋转
tf2::Quaternion q_ctl_rel = q_ctl_init.inverse() * q_ctl_current;
q_ctl_rel.normalize();

// 4. 将控制器坐标系的旋转转换到真实世界坐标系
// 使用旋转矩阵转换而不是欧拉角转换，避免万向节锁和顺序问题
// 定义从控制器坐标系到真实世界坐标系的旋转矩阵
// 根据坐标轴映射关系：控制器(x右,y上,z后) -> 真实世界(x前,y左,z上)
// 控制器x轴(右) -> 真实世界y轴负方向(左)
// 控制器y轴(上) -> 真实世界z轴(上)
// 控制器z轴(后) -> 真实世界x轴负方向(后)
tf2::Matrix3x3 R_ctl_to_real(
  0, 0, -1,  // 控制器x轴(1,0,0)映射到真实世界(0,0,-1) -> 但实际上是y-，见下面
  -1, 0, 0,  // 控制器y轴(0,1,0)映射到真实世界(-1,0,0) -> 但实际上是z+，见下面
  0, 1, 0   // 控制器z轴(0,0,1)映射到真实世界(0,1,0) -> 但实际上是x-，见下面
);
// 修正：上面的矩阵不对，我们需要仔细映射
// 实际上，控制器的基向量在真实世界中的表示为：
// 控制器x(1,0,0) -> 真实世界(0,-1,0) 即y-
// 控制器y(0,1,0) -> 真实世界(0,0,1) 即z+
// 控制器z(0,0,1) -> 真实世界(-1,0,0) 即x-
// 所以正确的旋转矩阵R_ctl_to_real为：
// [ 0,  0, -1 ]
// [-1,  0,  0 ]
// [ 0,  1,  0 ]

// 将控制器的相对旋转转换为旋转矩阵
tf2::Matrix3x3 R_ctl_rel(q_ctl_rel);

// 将控制器坐标系中的旋转矩阵转换到真实世界坐标系
// 公式: R_real_rel = R_ctl_to_real * R_ctl_rel * R_ctl_to_real.transpose()
tf2::Matrix3x3 R_real_rel = R_ctl_to_real * R_ctl_rel * R_ctl_to_real.transpose();

// 将旋转矩阵转换回四元数
tf2::Quaternion q_real_rel;
R_real_rel.getRotation(q_real_rel);
q_real_rel.normalize();

// 5. 叠加到当前的物体位置
geometry_msgs::msg::PoseStamped result;
result.header.stamp = this->now();
result.header.frame_id = "real";

// 计算新的位置
result.pose.position.x = l_real_pose.pose.position.x + dx_real;
result.pose.position.y = l_real_pose.pose.position.y + dy_real;
result.pose.position.z = l_real_pose.pose.position.z + dz_real;

// 计算新的方向
tf2::Quaternion q_real_current(
  l_real_pose.pose.orientation.x,
  l_real_pose.pose.orientation.y,
  l_real_pose.pose.orientation.z,
  l_real_pose.pose.orientation.w
);

// 旋转叠加：在全局坐标系中应用相对旋转
tf2::Quaternion q_real_new = q_real_rel * q_real_current;
q_real_new.normalize();

// 调试：检查旋转轴
// 获取四元数的旋转轴和角度
tf2::Vector3 axis_ctl = q_ctl_rel.getAxis();
double angle_ctl = q_ctl_rel.getAngle();

tf2::Vector3 axis_real = q_real_rel.getAxis();
double angle_real = q_real_rel.getAngle();

// 为了调试，也计算欧拉角
double roll_ctl, pitch_ctl, yaw_ctl;
tf2::Matrix3x3(q_ctl_rel).getRPY(roll_ctl, pitch_ctl, yaw_ctl);

double roll_real, pitch_real, yaw_real;
tf2::Matrix3x3(q_real_rel).getRPY(roll_real, pitch_real, yaw_real);

RCLCPP_INFO(this->get_logger(),
  "\033[1;33m[LEFT CTRL DEBUG]\033[0m\n"
  "┌──────────────────────────────────┬─────────────────────────────────────────────────────────┐\n"
  "│ Controller offset                │ dx:%9.6f  dy:%9.6f  dz:%9.6f │\n"
  "│ Transformed to real              │ dx:%9.6f  dy:%9.6f  dz:%9.6f │\n"
  "├──────────────────────────────────┼─────────────────────────────────────────────────────────┤\n"
  "│ Controller RPY (rad)            │ roll:%7.4f pitch:%7.4f yaw:%7.4f │\n"
  "│ Real world RPY (rad)            │ roll:%7.4f pitch:%7.4f yaw:%7.4f │\n"
  "├──────────────────────────────────┼─────────────────────────────────────────────────────────┤\n"
  "│ Controller rel rotation          │ axis:[%6.3f, %6.3f, %6.3f] angle:%6.3f │\n"
  "│ Real world rel rotation          │ axis:[%6.3f, %6.3f, %6.3f] angle:%6.3f │\n"
  "├──────────────────────────────────┼─────────────────────────────────────────────────────────┤\n"
  "│ Real world current pos           │ x:%9.6f  y:%9.6f  z:%9.6f │\n"
  "│ Real world new pos               │ x:%9.6f  y:%9.6f  z:%9.6f │\n"
  "│ Real world new quat              │ x:%9.6f  y:%9.6f  z:%9.6f  w:%9.6f │\n"
  "└──────────────────────────────────┴─────────────────────────────────────────────────────────┘",
  dx_ctl, dy_ctl, dz_ctl,
  dx_real, dy_real, dz_real,
  roll_ctl, pitch_ctl, yaw_ctl,
  roll_real, pitch_real, yaw_real,
  axis_ctl.x(), axis_ctl.y(), axis_ctl.z(), angle_ctl,
  axis_real.x(), axis_real.y(), axis_real.z(), angle_real,
  l_real_pose.pose.position.x, l_real_pose.pose.position.y, l_real_pose.pose.position.z,
  result.pose.position.x, result.pose.position.y, result.pose.position.z,
  q_real_new.x(), q_real_new.y(), q_real_new.z(), q_real_new.w()
);

result.pose.orientation.x = q_real_new.x();
result.pose.orientation.y = q_real_new.y();
result.pose.orientation.z = q_real_new.z();
result.pose.orientation.w = q_real_new.w();

l_joint_publisher_->publish(result);

























              }

              // // 求逆解
              // if (model_loaded_ && model_ptr_) {
              //   try {
              //     // 构建目标 SE3（注意 Eigen 四元数构造顺序：w,x,y,z）
              //     Eigen::Quaterniond quat_target(
              //       ps.pose.orientation.w,
              //       ps.pose.orientation.x,
              //       ps.pose.orientation.y,
              //       ps.pose.orientation.z
              //     );
              //     quat_target.normalize();
              //     SE3 oMdes(quat_target.toRotationMatrix(),
              //               Eigen::Vector3d(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z));
              //     // 将目标交给异步 IK 线程处理（避免在 90Hz 回调中阻塞）
              //     {
              //       std::lock_guard<std::mutex> lk(ik_mutex_);
              //       latest_oMdes_ = oMdes;
              //       // 使用 neutral 作为初始 guess，工作线程会用 last_solution_ 进行 warm start（若可用）
              //       latest_q_init_ = neutral(*model_ptr_);
              //       new_target_.store(true);
              //     }
              //     ik_cv_.notify_one();
              //   } catch (const std::exception & e) {
              //     RCLCPP_ERROR(this->get_logger(), "IK exception: %s", e.what());
              //   }
              // } else {
              //   RCLCPP_WARN(this->get_logger(), "Pinocchio model not loaded, cannot run IK.");
              // }
            }
            else {
              if (l_ctl_init) {
                l_ctl_init = false;
                RCLCPP_INFO(this->get_logger(), "\033[1;31mLeft trigger released, calling emergency stop service\033[0m");
                auto request = std::make_shared<std_srvs::srv::Empty::Request>();
                emergency_stop_client_->async_send_request(request);
                RCLCPP_INFO(this->get_logger(), "\033[1;31mEmergency stop service called\033[0m");
              }
            }

            if (custom_msg.right_controller.trigger == 1.0f) {
              
            }

            if (custom_msg.left_controller.gripper == 1.0f && left_gripper == false) {
              auto message = sensor_msgs::msg::JointState();
              message.header.stamp = this->now();
              message.header.frame_id = "base_link";
              
              float gripper_value = 0.05f;
              message.name = {"gripper"};
              message.position = {gripper_value};
              message.velocity = {0.0};  // 使用空数组会导致错误，这里设为0.0
              message.effort = {1.0};
              g_joint_publisher_->publish(message);
              left_gripper = true;

              RCLCPP_INFO(this->get_logger(), "Left gripper: [%f]", gripper_value);
            }
            else if (custom_msg.left_controller.gripper != 1.0f && left_gripper == true){
              auto message = sensor_msgs::msg::JointState();
              message.header.stamp = this->now();
              message.header.frame_id = "base_link";
              
              float gripper_value = 0.0f;
              message.name = {"gripper"};
              message.position = {gripper_value};
              message.velocity = {0.0};  // 使用空数组会导致错误，这里设为0.0
              message.effort = {1.0};
              g_joint_publisher_->publish(message);
              left_gripper = false;

              RCLCPP_INFO(this->get_logger(), "Left gripper: [%f]", gripper_value);
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
  // 修改：将 pose_publisher_ 改为 l_joint_publisher_
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr l_joint_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr r_joint_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr g_joint_publisher_;

  // 最新左臂真机姿态和相关变量
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr l_real_pose_subscriber_;
  bool l_real_has_new_pose_ = false;

  // 紧急停止
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr emergency_stop_client_;


  bool left_gripper = false;

  bool l_ctl_init = false;
  geometry_msgs::msg::PoseStamped l_ctl_init_pose;
  geometry_msgs::msg::PoseStamped l_real_pose;


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


  Eigen::VectorXd joint_position_lower_limits_;
  Eigen::VectorXd joint_position_upper_limits_;
};


int main(int argc, char * argv[])
{


















//    // 3. 计算旋转变换
//   // 获取控制器初始和当前的四元数
//   // 180 90 0
//   tf2::Quaternion q_ctl_init(
//     0,
//     0,
//     0,
//     1
//   );
  
//   tf2::Quaternion q_ctl_current(
//     0,
//     0,
//     0.259,
//     0.966
//   );
  







// // 计算相对旋转：从初始到当前的旋转
// tf2::Quaternion q_ctl_rel = q_ctl_init.inverse() * q_ctl_current;
// q_ctl_rel.normalize();

// // 4. 将控制器坐标系的旋转转换到真实世界坐标系
// // 关键：这不是相似变换，而是坐标轴的重映射！
// // 控制器: 绕x+旋转 -> 真实世界: 绕y-旋转
// // 控制器: 绕y+旋转 -> 真实世界: 绕z+旋转
// // 控制器: 绕z+旋转 -> 真实世界: 绕x-旋转

// // 从四元数中提取欧拉角（更容易理解转换）
// double roll_ctl, pitch_ctl, yaw_ctl;
// tf2::Matrix3x3(q_ctl_rel).getRPY(roll_ctl, pitch_ctl, yaw_ctl);

// // 根据映射关系转换欧拉角：
// // 控制器roll(绕x) -> 真实世界pitch(绕y)，但方向相反
// // 控制器pitch(绕y) -> 真实世界yaw(绕z)
// // 控制器yaw(绕z) -> 真实世界roll(绕x)，但方向相反
// double roll_real = -yaw_ctl;   // 控制器z+旋转 -> 真实世界x-旋转
// double pitch_real = -roll_ctl; // 控制器x+旋转 -> 真实世界y-旋转
// double yaw_real = pitch_ctl;  // 控制器y+旋转 -> 真实世界z+旋转

// // 将转换后的欧拉角转换回四元数
// tf2::Quaternion q_real_rel;
// q_real_rel.setRPY(roll_real, pitch_real, yaw_real);
// q_real_rel.normalize();


// // 计算新的方向
// tf2::Quaternion q_real_current(
//   0,
//   0,
//   0,
//   1
// );

// // 旋转叠加：在全局坐标系中应用相对旋转
// tf2::Quaternion q_real_new = q_real_rel * q_real_current;
// q_real_new.normalize();

// // 调试：检查旋转轴
// // 获取四元数的旋转轴和角度
// tf2::Vector3 axis_ctl = q_ctl_rel.getAxis();
// double angle_ctl = q_ctl_rel.getAngle();

// tf2::Vector3 axis_real = q_real_rel.getAxis();
// double angle_real = q_real_rel.getAngle();

// printf(
//   "\033[1;33m[LEFT CTRL DEBUG]\033[0m\n"
//   "┌──────────────────────────────────┬─────────────────────────────────────────────────────────┐\n"
//   "│ Controller RPY (rad)            │ roll:%7.4f pitch:%7.4f yaw:%7.4f │\n"
//   "│ Real world RPY (rad)            │ roll:%7.4f pitch:%7.4f yaw:%7.4f │\n"
//   "├──────────────────────────────────┼─────────────────────────────────────────────────────────┤\n"
//   "│ Controller rel rotation          │ axis:[%6.3f, %6.3f, %6.3f] angle:%6.3f │\n"
//   "│ Real world rel rotation          │ axis:[%6.3f, %6.3f, %6.3f] angle:%6.3f │\n"
//   "├──────────────────────────────────┼─────────────────────────────────────────────────────────┤\n"
//   "│ Real world new quat              │ x:%9.6f  y:%9.6f  z:%9.6f  w:%9.6f │\n"
//   "└──────────────────────────────────┴─────────────────────────────────────────────────────────┘\n",
//   roll_ctl, pitch_ctl, yaw_ctl,
//   roll_real, pitch_real, yaw_real,
//   axis_ctl.x(), axis_ctl.y(), axis_ctl.z(), angle_ctl,
//   axis_real.x(), axis_real.y(), axis_real.z(), angle_real,
//   q_real_new.x(), q_real_new.y(), q_real_new.z(), q_real_new.w()
// );























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