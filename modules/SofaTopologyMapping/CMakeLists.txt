cmake_minimum_required(VERSION 2.8.12)
project(SofaTopologyMapping)

set(HEADER_FILES
    Bezier2MeshMechanicalMapping.h
    Bezier2MeshMechanicalMapping.inl
    Bezier2MeshTopologicalMapping.h
    BezierTetra2BezierTriangleTopologicalMapping.h
    CenterPointTopologicalMapping.h
    Edge2QuadTopologicalMapping.h
    Hexa2QuadTopologicalMapping.h
    Hexa2TetraTopologicalMapping.h
    IdentityTopologicalMapping.h
    Mesh2BezierTopologicalMapping.h
    Mesh2PointMechanicalMapping.h
    Mesh2PointMechanicalMapping.inl
    Mesh2PointTopologicalMapping.h
    Quad2TriangleTopologicalMapping.h
    SimpleTesselatedHexaTopologicalMapping.h
    SimpleTesselatedTetraMechanicalMapping.h
    SimpleTesselatedTetraMechanicalMapping.inl
    SimpleTesselatedTetraTopologicalMapping.h
    SubsetTopologicalMapping.h
    Tetra2TriangleTopologicalMapping.h
    Triangle2EdgeTopologicalMapping.h
    config.h
    initTopologyMapping.h
)

set(SOURCE_FILES
    Bezier2MeshMechanicalMapping.cpp
    Bezier2MeshTopologicalMapping.cpp
    BezierTetra2BezierTriangleTopologicalMapping.cpp
    CenterPointTopologicalMapping.cpp
    Edge2QuadTopologicalMapping.cpp
    Hexa2QuadTopologicalMapping.cpp
    Hexa2TetraTopologicalMapping.cpp
    IdentityTopologicalMapping.cpp
    Mesh2BezierTopologicalMapping.cpp
    Mesh2PointMechanicalMapping.cpp
    Mesh2PointTopologicalMapping.cpp
    Quad2TriangleTopologicalMapping.cpp
    SimpleTesselatedHexaTopologicalMapping.cpp
    SimpleTesselatedTetraMechanicalMapping.cpp
    SimpleTesselatedTetraTopologicalMapping.cpp
    SubsetTopologicalMapping.cpp
    Tetra2TriangleTopologicalMapping.cpp
    Triangle2EdgeTopologicalMapping.cpp
    initTopologyMapping.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TOPOLOGY_MAPPING")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaGeneral_install(${PROJECT_NAME})
