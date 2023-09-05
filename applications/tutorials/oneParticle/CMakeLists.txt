cmake_minimum_required(VERSION 3.12)
project(oneParticle)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.Component REQUIRED)
sofa_find_package(Sofa.GUI REQUIRED)

add_executable(${PROJECT_NAME} oneParticle.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Graph Sofa.Component Sofa.GUI )
