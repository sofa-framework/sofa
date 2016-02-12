cmake_minimum_required(VERSION 2.8.12)
project(SofaGTestMain)

## Dependencies
find_package(SofaSimulation REQUIRED)
find_package(GTest CONFIG REQUIRED)

## Version
set(SOFAGTESTMAIN_VERSION ${SofaSimulation_VERSION})

add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_library(SofaGTestMain SofaGTestMain.cpp)
target_link_libraries(SofaGTestMain gtest SofaSimulationTree SofaSimulationGraph SofaTest)

## Install rules for the library; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${SOFAGTESTMAIN_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
