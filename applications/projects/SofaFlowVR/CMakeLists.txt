cmake_minimum_required(VERSION 3.22)
project(SofaFlowVR)

find_package(SofaGui REQUIRED)

add_executable(${PROJECT_NAME} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui)
