include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
    DynamicSparseGridTopologyContainer.h
    DynamicSparseGridTopologyModifier.h
    MultilevelHexahedronSetTopologyContainer.h
    SparseGridMultipleTopology.h
    SparseGridRamificationTopology.h
    HexahedronCompositeFEMForceFieldAndMass.h
    HexahedronCompositeFEMForceFieldAndMass.inl
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
    HexahedronCompositeFEMForceFieldAndMass.cpp
    HexahedronCompositeFEMMapping.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_NON_UNIFORM_FEM" )
AddLinkerDependencies(SofaSimpleFem ${SOFA_VISUAL_LIB} SofaDenseSolver SofaVolumetricData )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
