cmake_minimum_required(VERSION 3.1)
project(SofaBoundaryCondition)

set(HEADER_FILES
    config.h
    initBoundaryCondition.h
    )

set(SOURCE_FILES
    initBoundaryCondition.cpp
    )

list(APPEND HEADER_FILES
    AffineMovementConstraint.h
    AffineMovementConstraint.inl
    ConicalForceField.h
    ConicalForceField.inl
    ConstantForceField.h
    ConstantForceField.inl
    DiagonalVelocityDampingForceField.h
    DiagonalVelocityDampingForceField.inl
    EdgePressureForceField.h
    EdgePressureForceField.inl
    EllipsoidForceField.h
    EllipsoidForceField.inl
    FixedConstraint.h
    FixedConstraint.inl
    FixedPlaneConstraint.h
    FixedPlaneConstraint.inl
    FixedRotationConstraint.h
    FixedRotationConstraint.inl
    FixedTranslationConstraint.h
    FixedTranslationConstraint.inl
    HermiteSplineConstraint.h
    HermiteSplineConstraint.inl
    LinearForceField.h
    LinearForceField.inl
    LinearMovementConstraint.h
    LinearMovementConstraint.inl
    LinearVelocityConstraint.h
    LinearVelocityConstraint.inl
    OscillatingTorsionPressureForceField.h
    OscillatingTorsionPressureForceField.inl
    OscillatorConstraint.h
    OscillatorConstraint.inl
    ParabolicConstraint.h
    ParabolicConstraint.inl
    PartialFixedConstraint.h
    PartialFixedConstraint.inl
    PartialLinearMovementConstraint.h
    PartialLinearMovementConstraint.inl
    PatchTestMovementConstraint.h
    PatchTestMovementConstraint.inl
    PlaneForceField.h
    PlaneForceField.inl
    PointConstraint.h
    PointConstraint.inl
    PositionBasedDynamicsConstraint.h
    PositionBasedDynamicsConstraint.inl
    QuadPressureForceField.h
    QuadPressureForceField.inl
    SkeletalMotionConstraint.h
    SkeletalMotionConstraint.inl
    SphereForceField.h
    SphereForceField.inl
    SurfacePressureForceField.h
    SurfacePressureForceField.inl
    TaitSurfacePressureForceField.h
    TaitSurfacePressureForceField.inl
    TorsionForceField.h
    TorsionForceField.inl
    TrianglePressureForceField.h
    TrianglePressureForceField.inl
    UniformVelocityDampingForceField.h
    UniformVelocityDampingForceField.inl
    ProjectToLineConstraint.h
    ProjectToLineConstraint.inl
    ProjectToPlaneConstraint.h
    ProjectToPlaneConstraint.inl
    ProjectToPointConstraint.h
    ProjectToPointConstraint.inl
    ProjectDirectionConstraint.h
    ProjectDirectionConstraint.inl
    )

list(APPEND SOURCE_FILES
    AffineMovementConstraint.cpp
    ConicalForceField.cpp
    ConstantForceField.cpp
    DiagonalVelocityDampingForceField.cpp
    EdgePressureForceField.cpp
    EllipsoidForceField.cpp
    FixedConstraint.cpp
    FixedPlaneConstraint.cpp
    FixedRotationConstraint.cpp
    FixedTranslationConstraint.cpp
    HermiteSplineConstraint.cpp
    LinearForceField.cpp
    LinearMovementConstraint.cpp
    LinearVelocityConstraint.cpp
    OscillatingTorsionPressureForceField.cpp
    OscillatorConstraint.cpp
    ParabolicConstraint.cpp
    PartialFixedConstraint.cpp
    PartialLinearMovementConstraint.cpp
    PatchTestMovementConstraint.cpp
    PlaneForceField.cpp
    PointConstraint.cpp
    PositionBasedDynamicsConstraint.cpp
    QuadPressureForceField.cpp
    SkeletalMotionConstraint.cpp
    SphereForceField.cpp
    SurfacePressureForceField.cpp
    TaitSurfacePressureForceField.cpp
    TorsionForceField.cpp
    TrianglePressureForceField.cpp
    UniformVelocityDampingForceField.cpp
    ProjectToLineConstraint.cpp
    ProjectToPlaneConstraint.cpp
    ProjectToPointConstraint.cpp
    ProjectDirectionConstraint.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    list(APPEND HEADER_FILES
        BuoyantForceField.h
        BuoyantForceField.inl
        VaccumSphereForceField.h
        VaccumSphereForceField.inl
        )
    list(APPEND SOURCE_FILES
        BuoyantForceField.cpp
        VaccumSphereForceField.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaEigen2Solver)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  "${SOFAGENERAL_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BOUNDARY_CONDITION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
