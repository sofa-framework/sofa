cmake_minimum_required(VERSION 3.12)
project(sofaInfo)

find_package(Sofa.Config)
sofa_find_package(Sofa.Component)
sofa_find_package(Sofa.Simulation.Graph)

add_executable(${PROJECT_NAME} sofaInfo.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.Component Sofa.Simulation.Graph)
