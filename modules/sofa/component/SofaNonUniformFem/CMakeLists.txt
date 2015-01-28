include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initNonUniformFEM.h 
    # ../forcefield/NonUniformHexahedralFEMForceFieldAndMass.h 
    # ../forcefield/NonUniformHexahedralFEMForceFieldAndMass.inl 
    # ../forcefield/NonUniformHexahedronFEMForceFieldAndMass.h 
    # ../forcefield/NonUniformHexahedronFEMForceFieldAndMass.inl 
#    ../forcefield/NonUniformHexahedronFEMForceFieldDensity.h 
#    ../forcefield/NonUniformHexahedronFEMForceFieldDensity.inl 
    # ../topology/DynamicSparseGridGeometryAlgorithms.h 
    # ../topology/DynamicSparseGridGeometryAlgorithms.inl 
    # ../topology/DynamicSparseGridTopologyAlgorithms.h 
    # ../topology/DynamicSparseGridTopologyAlgorithms.inl 
    # ../topology/DynamicSparseGridTopologyContainer.h 
    # ../topology/DynamicSparseGridTopologyModifier.h 
    # ../topology/MultilevelHexahedronSetTopologyContainer.h 
    # ../topology/SparseGridMultipleTopology.h 
    # ../topology/SparseGridRamificationTopology.h 
    # ../forcefield/HexahedronCompositeFEMForceFieldAndMass.h 
    # ../forcefield/HexahedronCompositeFEMForceFieldAndMass.inl 
    # ../mapping/HexahedronCompositeFEMMapping.h 
    # ../mapping/HexahedronCompositeFEMMapping.inl
    )
    
set(SOURCE_FILES

     ../initNonUniformFEM.cpp
#     ../forcefield/NonUniformHexahedralFEMForceFieldAndMass.cpp 
#     ../forcefield/NonUniformHexahedronFEMForceFieldAndMass.cpp 
# #    ../forcefield/NonUniformHexahedronFEMForceFieldDensity.cpp 
#     ../topology/MultilevelHexahedronSetTopologyContainer.cpp 
#     ../topology/DynamicSparseGridGeometryAlgorithms.cpp 
#     ../topology/DynamicSparseGridTopologyAlgorithms.cpp 
#     ../topology/DynamicSparseGridTopologyContainer.cpp 
#     ../topology/DynamicSparseGridTopologyModifier.cpp 
#     ../topology/SparseGridMultipleTopology.cpp 
#     ../topology/SparseGridRamificationTopology.cpp 
#     ../forcefield/HexahedronCompositeFEMForceFieldAndMass.cpp 
#     ../mapping/HexahedronCompositeFEMMapping.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_NON_UNIFORM_FEM" )
AddLinkerDependencies(SofaSimpleFem ${SOFA_VISUAL_LIB} SofaDenseSolver SofaVolumetricData )
        
include(${SOFA_CMAKE_DIR}/postProject.cmake)
