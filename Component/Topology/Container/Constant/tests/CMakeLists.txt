cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Topology.Container.Constant_test)

set(SOURCE_FILES
    MeshTopology_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Container.Constant)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
