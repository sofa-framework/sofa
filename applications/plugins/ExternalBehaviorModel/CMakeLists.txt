cmake_minimum_required(VERSION 3.22)
project(ExternalBehaviorModel VERSION 0.1)


set(HEADER_FILES
    FEMGridBehaviorModel.h
    FEMGridBehaviorModel.inl
    config.h
)

set(SOURCE_FILES
    FEMGridBehaviorModel.cpp
    initExternalBehaviorModel.cpp
)


sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)
sofa_find_package(Sofa.Component.Topology REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(SofaGraphComponent REQUIRED)
sofa_find_package(Sofa.Component.ODESolver.Backward REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Component.LinearSolver.Iterative Sofa.Component.Topology Sofa.Component.Constraint.Projective SofaGraphComponent Sofa.Component.ODESolver.Backward Sofa.Component.SolidMechanics.FEM.Elastic)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_${PROJECT_NAME}")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
