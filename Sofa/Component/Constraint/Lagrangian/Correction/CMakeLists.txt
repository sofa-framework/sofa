cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Constraint.Lagrangian.Correction LANGUAGES CXX)

set(SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR "src/sofa/component/constraint/lagrangian/correction")

set(HEADER_FILES
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/init.h
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/GenericConstraintCorrection.h
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/LinearSolverConstraintCorrection.h
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/LinearSolverConstraintCorrection.inl
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/PrecomputedConstraintCorrection.h
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/PrecomputedConstraintCorrection.inl
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/UncoupledConstraintCorrection.h
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/UncoupledConstraintCorrection.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/GenericConstraintCorrection.cpp
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/LinearSolverConstraintCorrection.cpp
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/PrecomputedConstraintCorrection.cpp
    ${SOFACOMPONENTCONSTRAINTLAGRANGIANCORRECTION_SOURCE_DIR}/UncoupledConstraintCorrection.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Mass REQUIRED) # UncoupledCC needs UniformMass
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED) # PrecomputedCC needs CGLinearSolver
sofa_find_package(Sofa.Component.ODESolver.Backward REQUIRED) # PrecomputedCC needs EulerSolver

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mass)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.LinearSolver.Iterative)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.ODESolver.Backward)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
