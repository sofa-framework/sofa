cmake_minimum_required(VERSION 3.1)
project(SofaBaseMechanics)

set(HEADER_FILES

    BarycentricMappers/BarycentricMapper.h
    BarycentricMappers/BarycentricMapper.inl
    BarycentricMappers/TopologyBarycentricMapper.h
    BarycentricMappers/TopologyBarycentricMapper.inl
    BarycentricMappers/BarycentricMapperMeshTopology.h
    BarycentricMappers/BarycentricMapperMeshTopology.inl
    BarycentricMappers/BarycentricMapperRegularGridTopology.h
    BarycentricMappers/BarycentricMapperRegularGridTopology.inl
    BarycentricMappers/BarycentricMapperSparseGridTopology.h
    BarycentricMappers/BarycentricMapperSparseGridTopology.inl
    BarycentricMappers/BarycentricMapperTopologyContainer.h
    BarycentricMappers/BarycentricMapperTopologyContainer.inl
    BarycentricMappers/BarycentricMapperEdgeSetTopology.h
    BarycentricMappers/BarycentricMapperEdgeSetTopology.inl
    BarycentricMappers/BarycentricMapperTriangleSetTopology.h
    BarycentricMappers/BarycentricMapperTriangleSetTopology.inl
    BarycentricMappers/BarycentricMapperQuadSetTopology.h
    BarycentricMappers/BarycentricMapperQuadSetTopology.inl
    BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h
    BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl
    BarycentricMappers/BarycentricMapperHexahedronSetTopology.h
    BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl

    AddMToMatrixFunctor.h
    BarycentricMapping.h
    BarycentricMapping.inl
    DiagonalMass.h
    DiagonalMass.inl
    IdentityMapping.h
    IdentityMapping.inl
    MappedObject.h
    MappedObject.inl
    MechanicalObject.h
    MechanicalObject.inl
    SubsetMapping.h
    SubsetMapping.inl
    UniformMass.h
    UniformMass.inl
    config.h
    initBaseMechanics.h
)

set(SOURCE_FILES

    BarycentricMappers/BarycentricMapper.cpp
    BarycentricMappers/TopologyBarycentricMapper.cpp
    BarycentricMappers/BarycentricMapperMeshTopology.cpp
    BarycentricMappers/BarycentricMapperRegularGridTopology.cpp
    BarycentricMappers/BarycentricMapperSparseGridTopology.cpp
    BarycentricMappers/BarycentricMapperTopologyContainer.cpp
    BarycentricMappers/BarycentricMapperEdgeSetTopology.cpp
    BarycentricMappers/BarycentricMapperTriangleSetTopology.cpp
    BarycentricMappers/BarycentricMapperQuadSetTopology.cpp
    BarycentricMappers/BarycentricMapperTetrahedronSetTopology.cpp
    BarycentricMappers/BarycentricMapperHexahedronSetTopology.cpp

    BarycentricMapping.cpp
    DiagonalMass.cpp
    IdentityMapping.cpp
    MappedObject.cpp
    MechanicalObject.cpp
    SubsetMapping.cpp
    UniformMass.cpp
    initBaseMechanics.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_MECHANICS")

## Will create duplicates of SofaBaseMechanics/BarycentricMappers/*.{h,inl} in SofaBaseMechanics/
sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
