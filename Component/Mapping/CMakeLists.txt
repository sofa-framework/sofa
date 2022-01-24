cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Mapping LANGUAGES CXX)

set(SOFACOMPONENTMAPPING_SOURCE_DIR "src/sofa/component/mapping")

set(HEADER_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidRigidMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidRigidMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/LineSetSkinningMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/LineSetSkinningMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SkinningMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SkinningMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/Mesh2PointMechanicalMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/Mesh2PointMechanicalMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappingRigid.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappingRigid.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BeamLinearMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BeamLinearMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMappingOperation.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMulti2Mapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMulti2Mapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMultiMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMultiMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DeformableOnRigidFrameMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DeformableOnRigidFrameMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceFromTargetMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceFromTargetMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMultiMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMultiMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareDistanceMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareDistanceMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMultiMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMultiMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/TubularMapping.h
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/TubularMapping.inl
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/VoidMapping.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapper.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/TopologyBarycentricMapper.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperMeshTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperRegularGridTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperSparseGridTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTopologyContainer.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperEdgeSetTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTriangleSetTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperQuadSetTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/RigidRigidMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/LineSetSkinningMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SkinningMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/Mesh2PointMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SimpleTesselatedTetraMechanicalMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BarycentricMappingRigid.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/BeamLinearMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMulti2Mapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/CenterOfMassMultiMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DeformableOnRigidFrameMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceFromTargetMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/DistanceMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/IdentityMultiMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareDistanceMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SquareMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/SubsetMultiMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/TubularMapping.cpp
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/VoidMapping.cpp
)

sofa_find_package(SofaSimulationCore REQUIRED)
sofa_find_package(Sofa.Component.Topology.Grid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Grid)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
