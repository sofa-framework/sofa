cmake_minimum_required(VERSION 3.12)
project(SofaConstraint)

set(SOFACONSTRAINT_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFACONSTRAINT_SRC}/config.h.in
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.h
    )

set(SOURCE_FILES
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.cpp
    )

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/BilateralConstraintResolution.h
    ${SOFACONSTRAINT_SRC}/BilateralInteractionConstraint.h
    ${SOFACONSTRAINT_SRC}/BilateralInteractionConstraint.inl
    ${SOFACONSTRAINT_SRC}/ConstraintAnimationLoop.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.inl
    ${SOFACONSTRAINT_SRC}/ConstraintSolverImpl.h
    ${SOFACONSTRAINT_SRC}/ConstraintStoreLambdaVisitor.h
    ${SOFACONSTRAINT_SRC}/ContactIdentifier.h
    ${SOFACONSTRAINT_SRC}/FreeMotionAnimationLoop.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.inl
    ${SOFACONSTRAINT_SRC}/GenericConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/GenericConstraintSolver.h
    ${SOFACONSTRAINT_SRC}/LCPConstraintSolver.h
    ${SOFACONSTRAINT_SRC}/LMDNewProximityIntersection.h
    ${SOFACONSTRAINT_SRC}/LMDNewProximityIntersection.inl
    ${SOFACONSTRAINT_SRC}/LinearSolverConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/LinearSolverConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.h
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.inl
    ${SOFACONSTRAINT_SRC}/MappingGeometricStiffnessForceField.h
    ${SOFACONSTRAINT_SRC}/MappingGeometricStiffnessForceField.inl
    ${SOFACONSTRAINT_SRC}/PrecomputedConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/PrecomputedConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/SlidingConstraint.h
    ${SOFACONSTRAINT_SRC}/SlidingConstraint.inl
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.h
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.inl
    ${SOFACONSTRAINT_SRC}/StopperConstraint.h
    ${SOFACONSTRAINT_SRC}/StopperConstraint.inl
    ${SOFACONSTRAINT_SRC}/UncoupledConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/UncoupledConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/UniformConstraint.h
    ${SOFACONSTRAINT_SRC}/UniformConstraint.inl
    ${SOFACONSTRAINT_SRC}/UnilateralInteractionConstraint.h
    ${SOFACONSTRAINT_SRC}/UnilateralInteractionConstraint.inl
    )
list(APPEND SOURCE_FILES
    ${SOFACONSTRAINT_SRC}/BilateralInteractionConstraint.cpp
    ${SOFACONSTRAINT_SRC}/ConstraintAnimationLoop.cpp
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.cpp
    ${SOFACONSTRAINT_SRC}/ConstraintSolverImpl.cpp
    ${SOFACONSTRAINT_SRC}/ConstraintStoreLambdaVisitor.cpp
    ${SOFACONSTRAINT_SRC}/ContactIdentifier.cpp
    ${SOFACONSTRAINT_SRC}/FreeMotionAnimationLoop.cpp
    ${SOFACONSTRAINT_SRC}/FrictionContact.cpp
    ${SOFACONSTRAINT_SRC}/GenericConstraintCorrection.cpp
    ${SOFACONSTRAINT_SRC}/GenericConstraintSolver.cpp
    ${SOFACONSTRAINT_SRC}/LCPConstraintSolver.cpp
    ${SOFACONSTRAINT_SRC}/LMDNewProximityIntersection.cpp
    ${SOFACONSTRAINT_SRC}/LinearSolverConstraintCorrection.cpp
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.cpp
    ${SOFACONSTRAINT_SRC}/MappingGeometricStiffnessForceField.cpp
    ${SOFACONSTRAINT_SRC}/PrecomputedConstraintCorrection.cpp
    ${SOFACONSTRAINT_SRC}/SlidingConstraint.cpp
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.cpp
    ${SOFACONSTRAINT_SRC}/StopperConstraint.cpp
    ${SOFACONSTRAINT_SRC}/UncoupledConstraintCorrection.cpp
    ${SOFACONSTRAINT_SRC}/UniformConstraint.cpp
    ${SOFACONSTRAINT_SRC}/UnilateralInteractionConstraint.cpp
    )

find_package(SofaBase REQUIRED) # SofaBaseLinearSolver
find_package(SofaImplicitOdeSolver REQUIRED)  
find_package(SofaMeshCollision REQUIRED)
find_package(SofaSimpleFem REQUIRED)
find_package(SofaUserInteraction REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaSimpleFem SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFACONSTRAINT_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFACONSTRAINT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
