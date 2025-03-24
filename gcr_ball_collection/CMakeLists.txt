cmake_minimum_required(VERSION 3.8)
project(gcr_ball_collection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(gcr_obj_detect)

install(
  PROGRAMS 
    src/ball_collection_node.py
    src/ball_collection_bot_sort_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY src config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
