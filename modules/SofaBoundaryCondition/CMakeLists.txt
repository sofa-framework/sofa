cmake_minimum_required(VERSION 2.8.12)
project(SofaBoundaryCondition)

set(HEADER_FILES
    AffineMovementConstraint.h
    AffineMovementConstraint.inl
    BuoyantForceField.h
    BuoyantForceField.inl
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
    TrianglePressureForceField.h
    TrianglePressureForceField.inl
    UniformVelocityDampingForceField.h
    UniformVelocityDampingForceField.inl
    VaccumSphereForceField.h
    VaccumSphereForceField.inl
    config.h
    initBoundaryCondition.h
)

set(SOURCE_FILES
    AffineMovementConstraint.cpp
    BuoyantForceField.cpp
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
    TrianglePressureForceField.cpp
    UniformVelocityDampingForceField.cpp
    VaccumSphereForceField.cpp
    initBoundaryCondition.cpp
)

if(SOFA_HAVE_EIGEN2)

    list(APPEND HEADER_FILES
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
          ProjectToLineConstraint.cpp
          ProjectToPlaneConstraint.cpp
          ProjectToPointConstraint.cpp
          ProjectDirectionConstraint.cpp
    )

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
if(SOFA_HAVE_EIGEN2)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  "${SOFAGENERAL_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BOUNDARY_CONDITION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaGeneral_install(${PROJECT_NAME})
