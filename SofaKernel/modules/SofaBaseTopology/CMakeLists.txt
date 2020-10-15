cmake_minimum_required(VERSION 3.12)
project(SofaBaseTopology)


set(HEADER_FILES
    CommonAlgorithms.h
    EdgeSetGeometryAlgorithms.h
    EdgeSetGeometryAlgorithms.inl
    EdgeSetTopologyAlgorithms.h
    EdgeSetTopologyAlgorithms.inl
    EdgeSetTopologyContainer.h
    EdgeSetTopologyModifier.h
    GridTopology.h
    HexahedronSetGeometryAlgorithms.h
    HexahedronSetGeometryAlgorithms.inl
    HexahedronSetTopologyContainer.h
    HexahedronSetTopologyModifier.h
    MeshTopology.h
    NumericalIntegrationDescriptor.h
    NumericalIntegrationDescriptor.inl
    PointSetGeometryAlgorithms.h
    PointSetGeometryAlgorithms.inl
    PointSetTopologyAlgorithms.h
    PointSetTopologyAlgorithms.inl
    PointSetTopologyContainer.h
    PointSetTopologyModifier.h
    QuadSetGeometryAlgorithms.h
    QuadSetGeometryAlgorithms.inl
    QuadSetTopologyContainer.h
    QuadSetTopologyModifier.h
    RegularGridTopology.h
    SparseGridTopology.h
    TetrahedronSetGeometryAlgorithms.h
    TetrahedronSetGeometryAlgorithms.inl
    TetrahedronSetTopologyAlgorithms.h
    TetrahedronSetTopologyAlgorithms.inl
    TetrahedronSetTopologyContainer.h
    TetrahedronSetTopologyModifier.h
    TopologyData.h
    TopologyData.inl
    TopologyDataHandler.h
    TopologyDataHandler.inl
    TopologyEngine.h
    TopologyEngine.inl
    TopologySparseData.h
    TopologySparseData.inl
    TopologySparseDataHandler.h
    TopologySparseDataHandler.inl
    TopologySubsetData.h
    TopologySubsetData.inl
    TopologySubsetDataHandler.h
    TopologySubsetDataHandler.inl
    TriangleSetGeometryAlgorithms.h
    TriangleSetGeometryAlgorithms.inl
    TriangleSetTopologyAlgorithms.h
    TriangleSetTopologyAlgorithms.inl
    TriangleSetTopologyContainer.h
    TriangleSetTopologyModifier.h
    config.h
    initBaseTopology.h
)

set(SOURCE_FILES
    EdgeSetGeometryAlgorithms.cpp
    EdgeSetTopologyAlgorithms.cpp
    EdgeSetTopologyContainer.cpp
    EdgeSetTopologyModifier.cpp
    GridTopology.cpp
    HexahedronSetGeometryAlgorithms.cpp
    HexahedronSetTopologyContainer.cpp
    HexahedronSetTopologyModifier.cpp
    MeshTopology.cpp
    NumericalIntegrationDescriptor.cpp
    PointSetGeometryAlgorithms.cpp
    PointSetTopologyAlgorithms.cpp
    PointSetTopologyContainer.cpp
    PointSetTopologyModifier.cpp
    QuadSetGeometryAlgorithms.cpp
    QuadSetTopologyContainer.cpp
    QuadSetTopologyModifier.cpp
    RegularGridTopology.cpp
    SparseGridTopology.cpp
    TetrahedronSetGeometryAlgorithms.cpp
    TetrahedronSetTopologyAlgorithms.cpp
    TetrahedronSetTopologyContainer.cpp
    TetrahedronSetTopologyModifier.cpp
    TriangleSetGeometryAlgorithms.cpp
    TriangleSetTopologyAlgorithms.cpp
    TriangleSetTopologyContainer.cpp
    TriangleSetTopologyModifier.cpp
    initBaseTopology.cpp
)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaBase
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
