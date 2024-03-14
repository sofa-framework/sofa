cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.AnimationLoop LANGUAGES CXX)

set(SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR "src/sofa/component/animationloop")

set(HEADER_FILES
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/FreeMotionAnimationLoop.h
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/FreeMotionTask.h
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/ConstraintAnimationLoop.h
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/MultiStepAnimationLoop.h
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/MultiTagAnimationLoop.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/FreeMotionAnimationLoop.cpp
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/ConstraintAnimationLoop.cpp
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/MultiStepAnimationLoop.cpp
    ${SOFACOMPONENTANIMATIONLOOP_SOURCE_DIR}/MultiTagAnimationLoop.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
