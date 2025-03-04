cmake_minimum_required(VERSION 3.8)
project(gcr_obj_detect)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate ROS interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/YoloDetection.msg"
  "msg/YoloDetectionArray.msg"
  DEPENDENCIES builtin_interfaces
)

# Install Python modules
install(
  PROGRAMS 
    scripts/object_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY msg scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(builtin_interfaces)
ament_package()