cmake_minimum_required(VERSION 3.12)
project(SofaBaseTopology LANGUAGES CXX)

set(SOFABASETOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/config.h.in
    ${SOFABASETOPOLOGY_SRC}/fwd.h
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.h
    ${SOFABASETOPOLOGY_SRC}/CommonAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/GridTopology.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/MeshTopology.h
    ${SOFABASETOPOLOGY_SRC}/NumericalIntegrationDescriptor.h
    ${SOFABASETOPOLOGY_SRC}/NumericalIntegrationDescriptor.inl
    ${SOFABASETOPOLOGY_SRC}/PointSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/PointSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/RegularGridTopology.h
    ${SOFABASETOPOLOGY_SRC}/SparseGridTopology.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/TopologyData.h
    ${SOFABASETOPOLOGY_SRC}/TopologyData.inl
    ${SOFABASETOPOLOGY_SRC}/TopologyDataHandler.h
    ${SOFABASETOPOLOGY_SRC}/TopologyDataHandler.inl
    ${SOFABASETOPOLOGY_SRC}/TopologyEngine.h
    ${SOFABASETOPOLOGY_SRC}/TopologyEngine.inl
    ${SOFABASETOPOLOGY_SRC}/TopologySparseData.h
    ${SOFABASETOPOLOGY_SRC}/TopologySparseData.inl
    ${SOFABASETOPOLOGY_SRC}/TopologySparseDataHandler.h
    ${SOFABASETOPOLOGY_SRC}/TopologySparseDataHandler.inl
    ${SOFABASETOPOLOGY_SRC}/TopologySubsetData.h
    ${SOFABASETOPOLOGY_SRC}/TopologySubsetData.inl
    ${SOFABASETOPOLOGY_SRC}/TopologySubsetDataHandler.h
    ${SOFABASETOPOLOGY_SRC}/TopologySubsetDataHandler.inl
    ${SOFABASETOPOLOGY_SRC}/TriangleSetGeometryAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetGeometryAlgorithms.inl
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyAlgorithms.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyContainer.h
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyModifier.h
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/polygon_cube_intersection.h
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/vec.h
)

set(SOURCE_FILES
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.cpp
    ${SOFABASETOPOLOGY_SRC}/EdgeSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/EdgeSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/GridTopology.cpp
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/HexahedronSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/MeshTopology.cpp
    ${SOFABASETOPOLOGY_SRC}/NumericalIntegrationDescriptor.cpp
    ${SOFABASETOPOLOGY_SRC}/PointSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/PointSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/QuadSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/QuadSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/RegularGridTopology.cpp
    ${SOFABASETOPOLOGY_SRC}/SparseGridTopology.cpp
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/TetrahedronSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/TriangleSetGeometryAlgorithms.cpp
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyContainer.cpp
    ${SOFABASETOPOLOGY_SRC}/TriangleSetTopologyModifier.cpp
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    ${SOFABASETOPOLOGY_SRC}/polygon_cube_intersection/polygon_cube_intersection.cpp
)

find_package(SofaFramework REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASETOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASETOPOLOGY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
