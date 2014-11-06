cmake_minimum_required(VERSION 2.8)
project(SofaMiscMapping)

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

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimpleFem SofaRigid)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISCMAPPING_COMPILER_FLAGS "-DSOFA_BUILD_MISC_MAPPING")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISCMAPPING_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaMiscMapping_libraries
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
)
