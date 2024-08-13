cmake_minimum_required(VERSION 3.22)
project(meshconv)

find_package(MiniFlowVR REQUIRED)

add_executable(${PROJECT_NAME} meshconv.cpp tesselate.cpp)

target_link_libraries(${PROJECT_NAME} miniFlowVR)

