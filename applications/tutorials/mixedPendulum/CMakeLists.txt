cmake_minimum_required(VERSION 3.12)
project(mixedPendulum)

find_package(Sofa.GUI.Component REQUIRED)
find_package(Sofa.Component REQUIRED)

add_executable(${PROJECT_NAME} mixedPendulum.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Component Sofa.Component)
