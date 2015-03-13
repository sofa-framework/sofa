include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initMiscMapping.h
    BeamLinearMapping.h
    BeamLinearMapping.inl
    CenterPointMechanicalMapping.h
    CenterPointMechanicalMapping.inl
    CenterOfMassMapping.h
    CenterOfMassMapping.inl
    CenterOfMassMultiMapping.h
    CenterOfMassMultiMapping.inl
    CenterOfMassMulti2Mapping.h
    CenterOfMassMulti2Mapping.inl
    CurveMapping.h
    CurveMapping.inl
    DistanceMapping.h
    DistanceMapping.inl
    DistanceFromTargetMapping.h
    DistanceFromTargetMapping.inl
    ExternalInterpolationMapping.h
    ExternalInterpolationMapping.inl
    SubsetMultiMapping.h
    SubsetMultiMapping.inl
    TubularMapping.h
    TubularMapping.inl
    VoidMapping.h
    BarycentricMappingRigid.h
    BarycentricMappingRigid.inl
    DeformableOnRigidFrameMapping.h
    DeformableOnRigidFrameMapping.inl
    PCAOnRigidFrameMapping.h
    PCAOnRigidFrameMapping.inl
    CatmullRomSplineMapping.h
    CatmullRomSplineMapping.inl
    IdentityMultiMapping.h
    IdentityMultiMapping.inl

    )

set(SOURCE_FILES

    initMiscMapping.cpp
    BeamLinearMapping.cpp
    CenterPointMechanicalMapping.cpp
    CenterOfMassMapping.cpp
    CenterOfMassMultiMapping.cpp
    CenterOfMassMulti2Mapping.cpp
    CurveMapping.cpp
    DistanceMapping.cpp
    DistanceFromTargetMapping.cpp
    ExternalInterpolationMapping.cpp
    SubsetMultiMapping.cpp
    TubularMapping.cpp
    VoidMapping.cpp
    BarycentricMappingRigid.cpp
    DeformableOnRigidFrameMapping.cpp
    PCAOnRigidFrameMapping.cpp
    CatmullRomSplineMapping.cpp
    IdentityMultiMapping.cpp

    )

set(GROUP_BASE_DIR "..")

AddSourceDependencies(eigen)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_MAPPING" )
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaSimpleFem SofaRigid)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
