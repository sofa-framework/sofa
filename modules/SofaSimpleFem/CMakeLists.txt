cmake_minimum_required(VERSION 2.8)
project(SofaSimpleFem)

set(HEADER_FILES

    initSimpleFEM.h
    BeamFEMForceField.h
    BeamFEMForceField.inl
    HexahedralFEMForceField.h
    HexahedralFEMForceField.inl
    HexahedralFEMForceFieldAndMass.h
    HexahedralFEMForceFieldAndMass.inl
    HexahedronFEMForceField.h
    HexahedronFEMForceField.inl
    HexahedronFEMForceFieldAndMass.h
    HexahedronFEMForceFieldAndMass.inl
    TetrahedralCorotationalFEMForceField.h
    TetrahedralCorotationalFEMForceField.inl
    TetrahedronFEMForceField.h
    TetrahedronFEMForceField.inl
#    TriangularAnisotropicFEMForceField.h
#    TriangularAnisotropicFEMForceField.inl
#    TriangleFEMForceField.h
#    TriangleFEMForceField.inl
#    TriangularFEMForceField.h
#    TriangularFEMForceField.inl
    TriangularFEMForceFieldOptim.h
    TriangularFEMForceFieldOptim.inl
    PoissonContainer.h
    StiffnessContainer.h
    RadiusContainer.h
    LengthContainer.h
    )

set(SOURCE_FILES

    initSimpleFEM.cpp
    BeamFEMForceField.cpp
    HexahedralFEMForceField.cpp
    HexahedralFEMForceFieldAndMass.cpp
    HexahedronFEMForceField.cpp
    HexahedronFEMForceFieldAndMass.cpp
    TetrahedralCorotationalFEMForceField.cpp
    TetrahedronFEMForceField.cpp
#    TriangularAnisotropicFEMForceField.cpp
#    TriangleFEMForceField.cpp
#    TriangularFEMForceField.cpp
    TriangularFEMForceFieldOptim.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaBaseTopology)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFASIMPLEFEM_COMPILER_FLAGS "-DSOFA_BUILD_SIMPLE_FEM")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFASIMPLEFEM_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaCommon_libraries
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        )

install(DIRECTORY ${SofaSimpleFem_SOURCE_DIR}/ DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/SofaSimpleFem"
        COMPONENT SofaCommon_headers
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
