cmake_minimum_required(VERSION 2.8.12)
project(SofaEulerianFluid)

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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EULERIAN_FLUID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
