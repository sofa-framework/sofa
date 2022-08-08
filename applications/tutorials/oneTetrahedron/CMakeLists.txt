cmake_minimum_required(VERSION 3.12)
project(oneTetrahedron)

sofa_find_package(Sofa.GUI.Component REQUIRED)
sofa_find_package(Sofa.Component REQUIRED)

add_executable(${PROJECT_NAME} oneTetrahedron.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Component Sofa.Component)
