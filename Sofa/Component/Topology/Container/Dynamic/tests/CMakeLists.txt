cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Topology.Container.Dynamic_test)

set(SOURCE_FILES
    EdgeSetTopology_test.cpp
    HexahedronSetTopology_test.cpp
    PointSetTopology_test.cpp
    QuadSetTopology_test.cpp
    TetrahedronNumericalIntegration_test.cpp
    TetrahedronSetTopology_test.cpp
    TriangleNumericalIntegration_test.cpp
    TriangleSetTopology_test.cpp
    TriangleSubdividiers_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Topology.Container.Dynamic)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
