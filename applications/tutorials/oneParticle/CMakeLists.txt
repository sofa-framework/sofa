cmake_minimum_required(VERSION 3.12)
project(oneParticle)

find_package(Sofa.GUI.Component REQUIRED)
find_package(Sofa.Component REQUIRED)

add_executable(${PROJECT_NAME} oneParticle.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Component Sofa.Component)
