cmake_minimum_required(VERSION 3.1)
project(chainHybrid)

find_package(SofaGui)
find_package(SceneCreator)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain SceneCreator)
