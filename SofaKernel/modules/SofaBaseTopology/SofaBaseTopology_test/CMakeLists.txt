cmake_minimum_required(VERSION 3.12)

project(SofaBaseTopology_test)

find_package(SofaGeneralLoader REQUIRED)

set(SOURCE_FILES
    fake_TopologyScene.h
    fake_TopologyScene.cpp

    PointSetTopology_test.cpp
    EdgeSetTopology_test.cpp
    TriangleSetTopology_test.cpp
    QuadSetTopology_test.cpp
    TetrahedronSetTopology_test.cpp
    HexahedronSetTopology_test.cpp

    MeshTopology_test.cpp

    RegularGridTopology_test.cpp
    SparseGridTopology_test.cpp
    TetrahedronNumericalIntegration_test.cpp
    TriangleNumericalIntegration_test.cpp
    )

add_definitions("-DSOFABASETOPOLOGY_TEST_RESOURCES_DIR=\"${CMAKE_SOURCE_DIR}/share\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGTestMain SofaTest SofaGeneralLoader)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
