cmake_minimum_required(VERSION 3.8)
project(ur10e_perception_workcell)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch worlds config srdf scripts models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/perception_pick_and_place.py
    scripts/publish_planning_scene.py
    scripts/spawn_random_cube.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
