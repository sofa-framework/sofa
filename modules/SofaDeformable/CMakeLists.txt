cmake_minimum_required(VERSION 2.8)
project(SofaDeformable)

set(HEADER_FILES

    initDeformable.h
    FastTriangularBendingSprings.h
    FastTriangularBendingSprings.inl
    QuadularBendingSprings.h
    QuadularBendingSprings.inl
    RestShapeSpringsForceField.h
    RestShapeSpringsForceField.inl
    TriangularBendingSprings.h
    TriangularBendingSprings.inl
    TriangularBiquadraticSpringsForceField.h
    TriangularBiquadraticSpringsForceField.inl
    TriangularQuadraticSpringsForceField.h
    TriangularQuadraticSpringsForceField.inl
    TriangularTensorMassForceField.h
    TriangularTensorMassForceField.inl
    FrameSpringForceField.h
    FrameSpringForceField.inl
    MeshSpringForceField.h
    MeshSpringForceField.inl
    QuadBendingSprings.h
    QuadBendingSprings.inl
    RegularGridSpringForceField.h
    RegularGridSpringForceField.inl
    SpringForceField.h
    SpringForceField.inl
    StiffSpringForceField.h
    StiffSpringForceField.inl
    TriangleBendingSprings.h
    TriangleBendingSprings.inl
    VectorSpringForceField.h
    VectorSpringForceField.inl

    )

set(SOURCE_FILES

    initDeformable.cpp
    FastTriangularBendingSprings.cpp
    QuadularBendingSprings.cpp
    RestShapeSpringsForceField.cpp
    TriangularBendingSprings.cpp
    TriangularBiquadraticSpringsForceField.cpp
    TriangularQuadraticSpringsForceField.cpp
    TriangularTensorMassForceField.cpp
    FrameSpringForceField.cpp
    MeshSpringForceField.cpp
    QuadBendingSprings.cpp
    RegularGridSpringForceField.cpp
    SpringForceField.cpp
    StiffSpringForceField.cpp
    TriangleBendingSprings.cpp
    VectorSpringForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaBaseTopology)
#AddSourceDependencies(eigen)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFADEFORMABLE_COMPILER_FLAGS "-DSOFA_BUILD_DEFORMABLE")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFADEFORMABLE_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaCommon_libraries
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        )

install(DIRECTORY ${SofaDeformable_SOURCE_DIR}/ DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/SofaDeformable"
        COMPONENT SofaCommon_headers
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
