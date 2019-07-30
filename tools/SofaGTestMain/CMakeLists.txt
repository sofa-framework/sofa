cmake_minimum_required(VERSION 3.1)
project(SofaGTestMain)

## Dependencies
find_package(SofaSimulation REQUIRED)
find_package(GTest CONFIG REQUIRED)

## Version
set(SOFAGTESTMAIN_VERSION ${SofaSimulation_VERSION})

add_library(${PROJECT_NAME} SofaGTestMain.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaSimulationTree SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} PUBLIC gtest)

## Install rules for the library; CMake package configurations files
sofa_generate_package(NAME ${PROJECT_NAME} VERSION ${SOFAGTESTMAIN_VERSION} TARGETS ${PROJECT_NAME})
