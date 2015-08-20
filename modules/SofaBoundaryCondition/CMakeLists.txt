include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initBoundaryCondition.h
    BuoyantForceField.h
    BuoyantForceField.inl
    ConicalForceField.h
    ConicalForceField.inl
    ConstantForceField.h
    ConstantForceField.inl
    EdgePressureForceField.h
    EdgePressureForceField.inl
    EllipsoidForceField.h
    EllipsoidForceField.inl
    LinearForceField.h
    LinearForceField.inl
    OscillatingTorsionPressureForceField.h
    OscillatingTorsionPressureForceField.inl
    PlaneForceField.h
    PlaneForceField.inl
    QuadPressureForceField.h
    QuadPressureForceField.inl
    SphereForceField.h
    SphereForceField.inl
    SurfacePressureForceField.h
    SurfacePressureForceField.inl
    TaitSurfacePressureForceField.h
    TaitSurfacePressureForceField.inl
    TrianglePressureForceField.h
    TrianglePressureForceField.inl
    VaccumSphereForceField.h
    VaccumSphereForceField.inl
	UniformVelocityDampingForceField.h
	UniformVelocityDampingForceField.inl
	DiagonalVelocityDampingForceField.h
	DiagonalVelocityDampingForceField.inl
    AffineMovementConstraint.inl
    AffineMovementConstraint.h
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
     LinearMovementConstraint.h
    LinearMovementConstraint.inl
    LinearVelocityConstraint.h
    LinearVelocityConstraint.inl
    OscillatorConstraint.h
    OscillatorConstraint.inl
    ParabolicConstraint.h
    ParabolicConstraint.inl
    PartialFixedConstraint.h
    PartialFixedConstraint.inl
    PartialLinearMovementConstraint.h
    PartialLinearMovementConstraint.inl
    PatchTestMovementConstraint.inl
    PatchTestMovementConstraint.h
    PointConstraint.h
    PointConstraint.inl
    PositionBasedDynamicsConstraint.h
    PositionBasedDynamicsConstraint.inl
    SkeletalMotionConstraint.h
    SkeletalMotionConstraint.inl
    )

set(SOURCE_FILES

    initBoundaryCondition.cpp
    BuoyantForceField.cpp
    ConicalForceField.cpp
    ConstantForceField.cpp
    EdgePressureForceField.cpp
    EllipsoidForceField.cpp
    LinearForceField.cpp
    OscillatingTorsionPressureForceField.cpp
    PlaneForceField.cpp
    QuadPressureForceField.cpp
    SphereForceField.cpp
    SurfacePressureForceField.cpp
    TaitSurfacePressureForceField.cpp
    TrianglePressureForceField.cpp
    VaccumSphereForceField.cpp
	UniformVelocityDampingForceField.cpp
	DiagonalVelocityDampingForceField.cpp
    AffineMovementConstraint.cpp
    FixedConstraint.cpp
    FixedPlaneConstraint.cpp
    FixedRotationConstraint.cpp
    FixedTranslationConstraint.cpp
    HermiteSplineConstraint.cpp
    LinearMovementConstraint.cpp
    LinearVelocityConstraint.cpp
    OscillatorConstraint.cpp
    ParabolicConstraint.cpp
    PartialFixedConstraint.cpp
    PartialLinearMovementConstraint.cpp
    PatchTestMovementConstraint.cpp
    PointConstraint.cpp
    PositionBasedDynamicsConstraint.cpp
    SkeletalMotionConstraint.cpp
    )

if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
    list(APPEND HEADER_FILES "ProjectToLineConstraint.h")
    list(APPEND HEADER_FILES "ProjectToLineConstraint.inl")
    list(APPEND HEADER_FILES "ProjectToPlaneConstraint.h")
    list(APPEND HEADER_FILES "ProjectToPlaneConstraint.inl")
    list(APPEND HEADER_FILES "ProjectToPointConstraint.h")
    list(APPEND HEADER_FILES "ProjectToPointConstraint.inl")
    list(APPEND HEADER_FILES "ProjectDirectionConstraint.h")
    list(APPEND HEADER_FILES "ProjectDirectionConstraint.inl")

    list(APPEND SOURCE_FILES "ProjectToLineConstraint.cpp")
    list(APPEND SOURCE_FILES "ProjectToPlaneConstraint.cpp")
    list(APPEND SOURCE_FILES "ProjectToPointConstraint.cpp")
    list(APPEND SOURCE_FILES "ProjectDirectionConstraint.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BOUNDARY_CONDITION")
AddLinkerDependencies(SofaBaseTopology)
if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
    AddLinkerDependencies(SofaEigen2Solver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
