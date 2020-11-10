cmake_minimum_required(VERSION 3.12)
project(SofaBoundaryCondition LANGUAGES CXX)

set(SOFABOUNDARYCONDITION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/config.h.in
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.h
    )

set(SOURCE_FILES
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.cpp
    )

list(APPEND HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/AffineMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/AffineMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedPlaneConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedPlaneConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedRotationConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedRotationConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedTranslationConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedTranslationConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/HermiteSplineConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/HermiteSplineConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearVelocityConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearVelocityConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/OscillatorConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/OscillatorConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ParabolicConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ParabolicConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PartialFixedConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PartialFixedConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PartialLinearMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PartialLinearMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PatchTestMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PatchTestMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/PointConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PointConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PositionBasedDynamicsConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PositionBasedDynamicsConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SkeletalMotionConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/SkeletalMotionConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToLineConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToLineConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPlaneConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPlaneConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPointConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPointConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectDirectionConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectDirectionConstraint.inl
    )

list(APPEND SOURCE_FILES
    ${SOFABOUNDARYCONDITION_SRC}/AffineMovementConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/FixedConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/FixedPlaneConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/FixedRotationConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/FixedTranslationConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/HermiteSplineConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/LinearMovementConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/LinearVelocityConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/OscillatorConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ParabolicConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PartialFixedConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PartialLinearMovementConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PatchTestMovementConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PointConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PositionBasedDynamicsConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/SkeletalMotionConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToLineConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPlaneConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPointConstraint.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ProjectDirectionConstraint.cpp
    )

find_package(SofaBase REQUIRED) # SofaBaseTopology and SofaEigen2Solver

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaEigen2Solver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABOUNDARYCONDITION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABOUNDARYCONDITION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
