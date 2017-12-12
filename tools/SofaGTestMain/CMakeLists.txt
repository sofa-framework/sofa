cmake_minimum_required(VERSION 3.1)
project(SofaGTestMain)

## Dependencies
find_package(SofaSimulation REQUIRED)
find_package(GTest CONFIG REQUIRED)

## Version
set(SOFAGTESTMAIN_VERSION ${SofaSimulation_VERSION})

add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_library(SofaGTestMain SofaGTestMain.cpp)
target_link_libraries(SofaGTestMain SofaHelper SofaSimulationTree SofaSimulationGraph)

## Install rules for the library; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${SOFAGTESTMAIN_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
