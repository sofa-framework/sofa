cmake_minimum_required(VERSION 3.1)
project(SofaMiscMapping)

set(HEADER_FILES
    config.h
    initMiscMapping.h
    )

set(SOURCE_FILES
    initMiscMapping.cpp
    )

list(APPEND HEADER_FILES
    BarycentricMappingRigid.h
    BarycentricMappingRigid.inl
    BeamLinearMapping.h
    BeamLinearMapping.inl
    CenterOfMassMapping.h
    CenterOfMassMapping.inl
    CenterOfMassMulti2Mapping.h
    CenterOfMassMulti2Mapping.inl
    CenterOfMassMultiMapping.h
    CenterOfMassMultiMapping.inl
    DeformableOnRigidFrameMapping.h
    DeformableOnRigidFrameMapping.inl
    DistanceFromTargetMapping.h
    DistanceFromTargetMapping.inl
    DistanceMapping.h
    DistanceMapping.inl
    IdentityMultiMapping.h
    IdentityMultiMapping.inl
    SquareDistanceMapping.h
    SquareDistanceMapping.inl
    SquareMapping.h
    SquareMapping.inl
    SubsetMultiMapping.h
    SubsetMultiMapping.inl
    TubularMapping.h
    TubularMapping.inl
    VoidMapping.h
    )
list(APPEND SOURCE_FILES
    BarycentricMappingRigid.cpp
    BeamLinearMapping.cpp
    CenterOfMassMapping.cpp
    CenterOfMassMulti2Mapping.cpp
    CenterOfMassMultiMapping.cpp
    DeformableOnRigidFrameMapping.cpp
    DistanceFromTargetMapping.cpp
    DistanceMapping.cpp
    IdentityMultiMapping.cpp
    SquareDistanceMapping.cpp
    SquareMapping.cpp
    SubsetMultiMapping.cpp
    TubularMapping.cpp
    VoidMapping.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaRigid)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_MAPPING")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} "SofaMisc/${PROJECT_NAME}")
