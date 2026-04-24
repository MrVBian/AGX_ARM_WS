cmake_minimum_required(VERSION 3.8)
project(ik_solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# executable
add_executable(ik_solver_node src/main.cpp)
target_compile_features(ik_solver_node PUBLIC cxx_std_17)

# ament deps
ament_target_dependencies(ik_solver_node
  rclcpp
  ament_index_cpp
)

# link pinocchio / eigen robustly
if(TARGET pinocchio::pinocchio)
  target_link_libraries(ik_solver_node pinocchio::pinocchio)
elseif(TARGET pinocchio)
  target_link_libraries(ik_solver_node pinocchio)
elseif(DEFINED pinocchio_LIBRARIES)
  target_link_libraries(ik_solver_node ${pinocchio_LIBRARIES})
endif()

if(TARGET Eigen3::Eigen)
  target_link_libraries(ik_solver_node Eigen3::Eigen)
elseif(DEFINED EIGEN3_LIBS)
  target_link_libraries(ik_solver_node ${EIGEN3_LIBS})
endif()

# install
install(TARGETS ik_solver_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
