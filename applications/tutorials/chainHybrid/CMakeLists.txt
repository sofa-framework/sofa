cmake_minimum_required(VERSION 3.12)
project(chainHybrid)

find_package(SceneCreator REQUIRED)
find_package(Sofa.GUI REQUIRED)
find_package(Sofa.GUI.Component REQUIRED)
find_package(Sofa.Component REQUIRED)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME} SceneCreator Sofa.GUI.Component Sofa.Component SofaGui)
