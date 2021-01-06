cmake_minimum_required(VERSION 3.12)
project(SofaGTestMain)

## Dependencies
find_package(SofaSimulation REQUIRED)
find_package(GTest CONFIG REQUIRED)

## Version
set(SOFAGTESTMAIN_VERSION ${SofaSimulation_VERSION})

add_library(${PROJECT_NAME} SofaGTestMain.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} PUBLIC gtest)

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFAGTESTMAIN_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
