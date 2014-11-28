cmake_minimum_required(VERSION 2.8.11)
project(SofaNonUniformFem)
set(HEADER_FILES

    initNonUniformFEM.h
    NonUniformHexahedralFEMForceFieldAndMass.h
    NonUniformHexahedralFEMForceFieldAndMass.inl
    NonUniformHexahedronFEMForceFieldAndMass.h
    NonUniformHexahedronFEMForceFieldAndMass.inl
#    NonUniformHexahedronFEMForceFieldDensity.h
#    NonUniformHexahedronFEMForceFieldDensity.inl
    DynamicSparseGridGeometryAlgorithms.h
    DynamicSparseGridGeometryAlgorithms.inl
    DynamicSparseGridTopologyAlgorithms.h
#    DynamicSparseGridTopologyAlgorithms.inl
    DynamicSparseGridTopologyContainer.h
    DynamicSparseGridTopologyModifier.h
    MultilevelHexahedronSetTopologyContainer.h
    SparseGridMultipleTopology.h
    SparseGridRamificationTopology.h
    HexahedronCompositeFEMMapping.h
    HexahedronCompositeFEMMapping.inl
    )

set(SOURCE_FILES

    initNonUniformFEM.cpp
    NonUniformHexahedralFEMForceFieldAndMass.cpp
    NonUniformHexahedronFEMForceFieldAndMass.cpp
#    NonUniformHexahedronFEMForceFieldDensity.cpp
    MultilevelHexahedronSetTopologyContainer.cpp
    DynamicSparseGridGeometryAlgorithms.cpp
    DynamicSparseGridTopologyAlgorithms.cpp
    DynamicSparseGridTopologyContainer.cpp
    DynamicSparseGridTopologyModifier.cpp
    SparseGridMultipleTopology.cpp
    SparseGridRamificationTopology.cpp
    HexahedronCompositeFEMMapping.cpp
    )

if(SOFA_HAVE_NEWMAT)
  list(APPEND HEADER_FILES
    HexahedronCompositeFEMForceFieldAndMass.h
    HexahedronCompositeFEMForceFieldAndMass.inl
   
  )
  list(APPEND SOURCE_FILES
    HexahedronCompositeFEMForceFieldAndMass.cpp
  )
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaSimpleFem SofaBaseTopology SofaBaseMechanics)

if(SOFA_HAVE_NEWMAT)
  target_link_libraries( ${PROJECT_NAME} SofaDenseSolver ) 
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_DEPENDENCY_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_NON_UNIFORM_FEM")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAADVANCED_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaAdvancedTargets
        RUNTIME DESTINATION "${SOFAADVANCED_INSTALL_BIN_DIR}" COMPONENT SofaAdvanced_libraries
        LIBRARY DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        ARCHIVE DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        PUBLIC_HEADER DESTINATION "${SOFAADVANCED_INSTALL_INCLUDE_DIR}/SofaNonUniformFem" COMPONENT SofaAdvanced_headers
        )

