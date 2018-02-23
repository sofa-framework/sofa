cmake_minimum_required(VERSION 3.1)
project(SofaDeprecatedComponents)

## Dependencies
find_package(SofaFramework REQUIRED)
find_package(SofaGeneral REQUIRED)

add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} SofaHelper SofaUserInteraction)

## Install rules for the library; CMake package configurations files
sofa_create_package(${PROJECT_NAME} "1.0" ${PROJECT_NAME} ${PROJECT_NAME})
