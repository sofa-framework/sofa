cmake_minimum_required(VERSION 3.22)
project(SofaGTestMain)

## Dependencies
sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(GTest REQUIRED)

add_library(${PROJECT_NAME} SofaGTestMain.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.Simulation.Graph)
target_link_libraries(${PROJECT_NAME} PUBLIC GTest::gtest)

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Testing)
