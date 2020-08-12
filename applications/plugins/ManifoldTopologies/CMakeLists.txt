cmake_minimum_required(VERSION 3.12)
project(ManifoldTopologies VERSION 0.1)

set(HEADER_FILES
    ManifoldEdgeSetGeometryAlgorithms.h
    ManifoldEdgeSetGeometryAlgorithms.inl
    ManifoldEdgeSetTopologyAlgorithms.h
    ManifoldEdgeSetTopologyAlgorithms.inl
    ManifoldEdgeSetTopologyContainer.h
    ManifoldEdgeSetTopologyModifier.h
    ManifoldTetrahedronSetTopologyContainer.h
    ManifoldTopologyObject_double.h
    ManifoldTopologyObject_float.h
    ManifoldTriangleSetTopologyAlgorithms.h
    ManifoldTriangleSetTopologyAlgorithms.inl
    ManifoldTriangleSetTopologyContainer.h
    ManifoldTriangleSetTopologyModifier.h
    config.h
)

set(SOURCE_FILES
    ManifoldEdgeSetGeometryAlgorithms.cpp
    ManifoldEdgeSetTopologyAlgorithms.cpp
    ManifoldEdgeSetTopologyContainer.cpp
    ManifoldEdgeSetTopologyModifier.cpp
    ManifoldTetrahedronSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyAlgorithms.cpp
    ManifoldTriangleSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyModifier.cpp
    initManifoldTopologies.cpp
)

set(README_FILES ManifoldTopologies.txt)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
