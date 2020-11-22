cmake_minimum_required(VERSION 3.12)
project(SofaEulerianFluid VERSION 1.0)

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
#
#
## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
