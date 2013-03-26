cmake_minimum_required(VERSION 2.8)

project("SofaTopologyMapping")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initTopologyMapping.h 
    mapping/Mesh2PointMechanicalMapping.h 
    mapping/Mesh2PointMechanicalMapping.inl 
    mapping/SimpleTesselatedTetraMechanicalMapping.h 
    mapping/SimpleTesselatedTetraMechanicalMapping.inl 
    topology/CenterPointTopologicalMapping.h 
    topology/Edge2QuadTopologicalMapping.h 
    topology/Hexa2QuadTopologicalMapping.h 
    topology/Hexa2TetraTopologicalMapping.h 
    topology/Mesh2PointTopologicalMapping.h  
    topology/Quad2TriangleTopologicalMapping.h 
    topology/SimpleTesselatedHexaTopologicalMapping.h 
    topology/SimpleTesselatedTetraTopologicalMapping.h 
    topology/Tetra2TriangleTopologicalMapping.h 
    topology/Triangle2EdgeTopologicalMapping.h 
    topology/IdentityTopologicalMapping.h 
    topology/SubsetTopologicalMapping.h
           
    )
    
set(SOURCE_FILES


    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_TOPOLOGY_MAPPING")
set(LINKER_DEPENDENCIES SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/post.cmake)
