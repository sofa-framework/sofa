cmake_minimum_required(VERSION 3.1)
project(SofaMiscMapping)

set(HEADER_FILES
    BarycentricMappingRigid.h
    BarycentricMappingRigid.inl
    BeamLinearMapping.h
    BeamLinearMapping.inl
    CatmullRomSplineMapping.h
    CatmullRomSplineMapping.inl
    CenterOfMassMapping.h
    CenterOfMassMapping.inl
    CenterOfMassMulti2Mapping.h
    CenterOfMassMulti2Mapping.inl
    CenterOfMassMultiMapping.h
    CenterOfMassMultiMapping.inl
    CenterPointMechanicalMapping.h
    CenterPointMechanicalMapping.inl
    CurveMapping.h
    CurveMapping.inl
    DeformableOnRigidFrameMapping.h
    DeformableOnRigidFrameMapping.inl
    DistanceFromTargetMapping.h
    DistanceFromTargetMapping.inl
    DistanceMapping.h
    DistanceMapping.inl
    ExternalInterpolationMapping.h
    ExternalInterpolationMapping.inl
    IdentityMultiMapping.h
    IdentityMultiMapping.inl
    ProjectionToLineMapping.h
    ProjectionToLineMapping.inl
    ProjectionToPlaneMapping.h
    ProjectionToPlaneMapping.inl
    SquareDistanceMapping.h
    SquareDistanceMapping.inl
    SquareMapping.h
    SquareMapping.inl
    SubsetMultiMapping.h
    SubsetMultiMapping.inl
    TubularMapping.h
    TubularMapping.inl
    VoidMapping.h
    config.h
    initMiscMapping.h
)

set(SOURCE_FILES
    BarycentricMappingRigid.cpp
    BeamLinearMapping.cpp
    CatmullRomSplineMapping.cpp
    CenterOfMassMapping.cpp
    CenterOfMassMulti2Mapping.cpp
    CenterOfMassMultiMapping.cpp
    CenterPointMechanicalMapping.cpp
    CurveMapping.cpp
    DeformableOnRigidFrameMapping.cpp
    DistanceFromTargetMapping.cpp
    DistanceMapping.cpp
    ExternalInterpolationMapping.cpp
    IdentityMultiMapping.cpp
    ProjectionToLineMapping.cpp
    ProjectionToPlaneMapping.cpp
    SquareDistanceMapping.cpp
    SquareMapping.cpp
    SubsetMultiMapping.cpp
    TubularMapping.cpp
    VoidMapping.cpp
    initMiscMapping.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaRigid)


set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_MAPPING")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
