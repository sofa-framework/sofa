cmake_minimum_required(VERSION 2.8.11)
project(SofaBaseTopology)


set(HEADER_FILES

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
    TetrahedronSetTopologyAlgorithms.h
    TetrahedronSetTopologyAlgorithms.inl
    TetrahedronSetTopologyContainer.h
    BezierTetrahedronSetTopologyContainer.h
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
    TetrahedronSetTopologyAlgorithms.cpp
    TetrahedronSetTopologyContainer.cpp
    BezierTetrahedronSetTopologyContainer.cpp
    TetrahedronSetTopologyModifier.cpp
    TriangleSetGeometryAlgorithms.cpp
    TriangleSetTopologyAlgorithms.cpp
    TriangleSetTopologyContainer.cpp
    TriangleSetTopologyModifier.cpp

    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaSimulationCommon ) 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_DEPENDENCY_INCLUDE_DIR}>")

set( SOFABASETOPOLOGY_COMPILER_FLAGS "-DSOFA_BUILD_BASE_TOPOLOGY")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFABASETOPOLOGY_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaBaseTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}" COMPONENT SofaBase_libraries
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/SofaBaseTopology" COMPONENT SofaBase_headers
        )
