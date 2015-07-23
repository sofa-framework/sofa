include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initBaseTopology.h
    CommonAlgorithms.h
    CubeTopology.h
    CylinderGridTopology.h
    SphereQuadTopology.h
    EdgeSetGeometryAlgorithms.h
    EdgeSetGeometryAlgorithms.inl
    EdgeSetTopologyAlgorithms.h
    EdgeSetTopologyAlgorithms.inl
    EdgeSetTopologyContainer.h
    EdgeSetTopologyModifier.h
    GridTopology.h
    HexahedronSetGeometryAlgorithms.h
    HexahedronSetGeometryAlgorithms.inl
    HexahedronSetTopologyAlgorithms.h
    HexahedronSetTopologyAlgorithms.inl
    HexahedronSetTopologyContainer.h
    HexahedronSetTopologyModifier.h
	NumericalIntegrationDescriptor.h
	NumericalIntegrationDescriptor.inl
    MeshTopology.h
    PointSetGeometryAlgorithms.h
    PointSetGeometryAlgorithms.inl
    PointSetTopologyAlgorithms.h
    PointSetTopologyAlgorithms.inl
    PointSetTopologyContainer.h
    PointSetTopologyModifier.h
    QuadSetGeometryAlgorithms.h
    QuadSetGeometryAlgorithms.inl
    QuadSetTopologyAlgorithms.h
    QuadSetTopologyAlgorithms.inl
    QuadSetTopologyContainer.h
    QuadSetTopologyModifier.h
    RegularGridTopology.h
    SparseGridTopology.h
    TetrahedronSetGeometryAlgorithms.h
    TetrahedronSetGeometryAlgorithms.inl
    BezierTetrahedronSetGeometryAlgorithms.h
    BezierTetrahedronSetGeometryAlgorithms.inl
    BezierTriangleSetGeometryAlgorithms.h
    BezierTriangleSetGeometryAlgorithms.inl
    TetrahedronSetTopologyAlgorithms.h
    TetrahedronSetTopologyAlgorithms.inl
    TetrahedronSetTopologyContainer.h
    BezierTetrahedronSetTopologyContainer.h
    BezierTriangleSetTopologyContainer.h
    TetrahedronSetTopologyModifier.h
    TriangleSetGeometryAlgorithms.h
    TriangleSetGeometryAlgorithms.inl
    TriangleSetTopologyAlgorithms.h
    TriangleSetTopologyAlgorithms.inl
    TriangleSetTopologyContainer.h
    TriangleSetTopologyModifier.h
    TopologyData.h
    TopologyData.inl
    TopologyDataHandler.h
    TopologyDataHandler.inl
    TopologySparseDataHandler.h
    TopologySparseDataHandler.inl
    TopologySparseData.h
    TopologySparseData.inl
    TopologySubsetData.h
    TopologySubsetData.inl
    TopologySubsetDataHandler.h
    TopologySubsetDataHandler.inl
    TopologyEngine.h
    TopologyEngine.inl

    )

set(SOURCE_FILES

    initBaseTopology.cpp
    CubeTopology.cpp
    CylinderGridTopology.cpp
    SphereQuadTopology.cpp
    EdgeSetGeometryAlgorithms.cpp
    EdgeSetTopologyAlgorithms.cpp
    EdgeSetTopologyContainer.cpp
    EdgeSetTopologyModifier.cpp
    GridTopology.cpp
    HexahedronSetGeometryAlgorithms.cpp
    HexahedronSetTopologyAlgorithms.cpp
    HexahedronSetTopologyContainer.cpp
    HexahedronSetTopologyModifier.cpp
    MeshTopology.cpp
	NumericalIntegrationDescriptor.cpp
    PointSetGeometryAlgorithms.cpp
    PointSetTopologyAlgorithms.cpp
    PointSetTopologyContainer.cpp
    PointSetTopologyModifier.cpp
    QuadSetGeometryAlgorithms.cpp
    QuadSetTopologyAlgorithms.cpp
    QuadSetTopologyContainer.cpp
    QuadSetTopologyModifier.cpp
    RegularGridTopology.cpp
    SparseGridTopology.cpp
    TetrahedronSetGeometryAlgorithms.cpp
    BezierTetrahedronSetGeometryAlgorithms.cpp
	BezierTriangleSetGeometryAlgorithms.cpp
    TetrahedronSetTopologyAlgorithms.cpp
    TetrahedronSetTopologyContainer.cpp
    BezierTetrahedronSetTopologyContainer.cpp
	BezierTriangleSetTopologyContainer.cpp
    TetrahedronSetTopologyModifier.cpp
    TriangleSetGeometryAlgorithms.cpp
    TriangleSetTopologyAlgorithms.cpp
    TriangleSetTopologyContainer.cpp
    TriangleSetTopologyModifier.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_TOPOLOGY")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
