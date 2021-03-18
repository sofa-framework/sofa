cmake_minimum_required(VERSION 3.12)
project(SofaGTestMain)

## Dependencies
sofa_find_package(SofaSimulationGraph REQUIRED)
sofa_find_package(GTest CONFIG REQUIRED)

add_library(${PROJECT_NAME} SofaGTestMain.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} PUBLIC gtest)

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Testing)
