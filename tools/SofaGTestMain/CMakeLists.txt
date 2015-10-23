cmake_minimum_required(VERSION 2.8.12)

find_package(SofaSimulation REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_library(SofaGTestMain SofaGTestMain.cpp)
target_link_libraries(SofaGTestMain gtest SofaSimulationTree SofaSimulationGraph)
sofa_install_targets(SofaSimulation SofaGTestMain "")
