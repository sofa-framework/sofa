cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Topology.Container.Grid_test)

set(SOURCE_FILES
    CylinderGridTopology_test.cpp
    RegularGridTopology_test.cpp
    SparseGridTopology_test.cpp
    SphereGridTopology_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Topology.Container.Grid Sofa.Component.IO.Mesh)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
