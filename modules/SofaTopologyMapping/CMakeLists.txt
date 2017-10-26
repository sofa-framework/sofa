cmake_minimum_required(VERSION 3.1)
project(SofaTopologyMapping)

set(HEADER_FILES
    CenterPointTopologicalMapping.h
    Edge2QuadTopologicalMapping.h
    Hexa2QuadTopologicalMapping.h
    Hexa2TetraTopologicalMapping.h
    IdentityTopologicalMapping.h
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
    CenterPointTopologicalMapping.cpp
    Edge2QuadTopologicalMapping.cpp
    Hexa2QuadTopologicalMapping.cpp
    Hexa2TetraTopologicalMapping.cpp
    IdentityTopologicalMapping.cpp
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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TOPOLOGY_MAPPING")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
