cmake_minimum_required(VERSION 3.22)
project(Sofa.Component LANGUAGES CXX)

set(SOFACOMPONENT_SOURCE_DIR "src/sofa/component")

sofa_add_subdirectory_modules(SOFACOMPONENT_TARGETS
    DIRECTORIES
    ODESolver
    IO
    Playback
    SceneUtility
    Topology
    Visual
    LinearSystem
    LinearSolver
    Mass
    Diffusion
    Mapping
    StateContainer
    SolidMechanics
    Constraint
    AnimationLoop
    MechanicalLoad
    Collision
    Setting
    Controller
    Engine
    Haptics
)

set(HEADER_FILES
    ${SOFACOMPONENT_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENT_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENT_SOURCE_DIR}/init.cpp
)

set(TARGET_NAME ${PROJECT_NAME})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC ${SOFACOMPONENT_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_BUILD_TESTS)
    add_subdirectory(test)
endif()
