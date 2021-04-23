cmake_minimum_required(VERSION 3.12)
project(chainHybrid)

find_package(SceneCreator REQUIRED)
find_package(SofaGui REQUIRED)
find_package(SofaComponentAll REQUIRED)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME} SceneCreator SofaGui SofaComponentAll)
