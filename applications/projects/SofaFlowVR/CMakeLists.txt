cmake_minimum_required(VERSION 3.12)
project(SofaFlowVR)

find_package(SofaGui)

add_executable(${PROJECT_NAME} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
