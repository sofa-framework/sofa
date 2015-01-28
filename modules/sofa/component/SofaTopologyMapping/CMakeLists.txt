include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initTopologyMapping.h 
    # ../mapping/Mesh2PointMechanicalMapping.h 
    # ../mapping/Mesh2PointMechanicalMapping.inl 
    # ../mapping/SimpleTesselatedTetraMechanicalMapping.h 
    # ../mapping/SimpleTesselatedTetraMechanicalMapping.inl 
    # ../topology/CenterPointTopologicalMapping.h 
    # ../topology/Edge2QuadTopologicalMapping.h 
    # ../topology/Hexa2QuadTopologicalMapping.h 
     ../topology/Hexa2TetraTopologicalMapping.h
    # ../topology/Mesh2PointTopologicalMapping.h  
    # ../topology/Mesh2BezierTopologicalMapping.h  
    # ../topology/Quad2TriangleTopologicalMapping.h 
    # ../topology/SimpleTesselatedHexaTopologicalMapping.h 
    # ../topology/SimpleTesselatedTetraTopologicalMapping.h 
    # ../topology/Tetra2TriangleTopologicalMapping.h 
    # ../topology/Triangle2EdgeTopologicalMapping.h 
    # ../topology/IdentityTopologicalMapping.h 
    # ../topology/SubsetTopologicalMapping.h
           
    )
    
set(SOURCE_FILES

    ../initTopologyMapping.cpp 
    # ../mapping/Mesh2PointMechanicalMapping.cpp 
    # ../mapping/SimpleTesselatedTetraMechanicalMapping.cpp 
    # ../topology/CenterPointTopologicalMapping.cpp 
    # ../topology/Edge2QuadTopologicalMapping.cpp 
    # ../topology/Hexa2QuadTopologicalMapping.cpp 
     ../topology/Hexa2TetraTopologicalMapping.cpp
    # ../topology/Mesh2PointTopologicalMapping.cpp  
    # ../topology/Mesh2BezierTopologicalMapping.cpp  
    # ../topology/Quad2TriangleTopologicalMapping.cpp 
    # ../topology/SimpleTesselatedHexaTopologicalMapping.cpp 
    # ../topology/SimpleTesselatedTetraTopologicalMapping.cpp 
    # ../topology/Tetra2TriangleTopologicalMapping.cpp 
    # ../topology/Triangle2EdgeTopologicalMapping.cpp 
    # ../topology/IdentityTopologicalMapping.cpp 
    # ../topology/SubsetTopologicalMapping.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_TOPOLOGY_MAPPING")
AddLinkerDependencies(SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
