cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Mapping.Linear LANGUAGES CXX)

set(SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR "src/sofa/component/mapping/linear")

set(HEADER_FILES
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/init.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappingRigid.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappingRigid.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BeamLinearMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BeamLinearMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMappingOperation.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMulti2Mapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMulti2Mapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMultiMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMultiMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/DeformableOnRigidFrameMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/DeformableOnRigidFrameMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMultiMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMultiMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/LinearMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/LineSetSkinningMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/LineSetSkinningMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointTopologicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointTopologicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedHexaTopologicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraTopologicalMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SkinningMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SkinningMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMultiMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMultiMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/TubularMapping.h
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/TubularMapping.inl
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/VoidMapping.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BarycentricMappingRigid.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/BeamLinearMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMulti2Mapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/CenterOfMassMultiMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/DeformableOnRigidFrameMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/IdentityMultiMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/LineSetSkinningMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/Mesh2PointTopologicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedHexaTopologicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SimpleTesselatedTetraTopologicalMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SkinningMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/SubsetMultiMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/TubularMapping.cpp
    ${SOFACOMPONENTMAPPINGLINEAR_SOURCE_DIR}/VoidMapping.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Dynamic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Grid)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_MAPPING_LINEAR_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_MAPPING_LINEAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
