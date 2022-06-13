cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Haptic LANGUAGES CXX)

set(SOFACOMPONENTHAPTIC_SOURCE_DIR "src/sofa/component/haptic")

set(HEADER_FILES
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/init.h
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/ForceFeedback.h
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/LCPForceFeedback.h
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/LCPForceFeedback.inl
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/MechanicalStateForceFeedback.h
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/NullForceFeedback.h
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/NullForceFeedbackT.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/ForceFeedback.cpp
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/LCPForceFeedback.cpp
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/NullForceFeedback.cpp
    ${SOFACOMPONENTHAPTIC_SOURCE_DIR}/NullForceFeedbackT.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Solver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_HAPTIC_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_HAPTIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
