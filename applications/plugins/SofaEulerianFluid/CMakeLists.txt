cmake_minimum_required(VERSION 3.1)
project(SofaEulerianFluid)

set(SOFAEULERIANFLUID_VERSION 1.0)

set(HEADER_FILES
    Fluid2D.h
    Fluid3D.h
    Grid2D.h
    Grid3D.h
    config.h
    initEulerianFluid.h
)

set(SOURCE_FILES
    Fluid2D.cpp
    Fluid3D.cpp
    Grid2D.cpp
    Grid3D.cpp
    initEulerianFluid.cpp
)

set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
#target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EULERIAN_FLUID -DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

## Install rules for the library and headers; CMake package configurations files
#sofa_create_package(SofaEulerianFluid ${SOFATEST_VERSION} ${PROJECT_NAME} SofaEulerianFluid)

