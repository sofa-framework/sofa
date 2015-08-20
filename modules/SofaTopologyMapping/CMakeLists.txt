include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initTopologyMapping.h
    Mesh2PointMechanicalMapping.h
    Mesh2PointMechanicalMapping.inl
    SimpleTesselatedTetraMechanicalMapping.h
    SimpleTesselatedTetraMechanicalMapping.inl
    CenterPointTopologicalMapping.h
    Edge2QuadTopologicalMapping.h
    Hexa2QuadTopologicalMapping.h
    Hexa2TetraTopologicalMapping.h
    Mesh2PointTopologicalMapping.h
    Mesh2BezierTopologicalMapping.h
    Quad2TriangleTopologicalMapping.h
    SimpleTesselatedHexaTopologicalMapping.h
    SimpleTesselatedTetraTopologicalMapping.h
    Tetra2TriangleTopologicalMapping.h
    Triangle2EdgeTopologicalMapping.h
    IdentityTopologicalMapping.h
    SubsetTopologicalMapping.h

    )

set(SOURCE_FILES

    initTopologyMapping.cpp
    Mesh2PointMechanicalMapping.cpp
    SimpleTesselatedTetraMechanicalMapping.cpp
    CenterPointTopologicalMapping.cpp
    Edge2QuadTopologicalMapping.cpp
    Hexa2QuadTopologicalMapping.cpp
    Hexa2TetraTopologicalMapping.cpp
    Mesh2PointTopologicalMapping.cpp
    Mesh2BezierTopologicalMapping.cpp
    Quad2TriangleTopologicalMapping.cpp
    SimpleTesselatedHexaTopologicalMapping.cpp
    SimpleTesselatedTetraTopologicalMapping.cpp
    Tetra2TriangleTopologicalMapping.cpp
    Triangle2EdgeTopologicalMapping.cpp
    IdentityTopologicalMapping.cpp
    SubsetTopologicalMapping.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_TOPOLOGY_MAPPING")
AddLinkerDependencies(SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
