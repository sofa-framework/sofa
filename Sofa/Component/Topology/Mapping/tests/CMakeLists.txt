cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Topology.Mapping_test)

set(SOURCE_FILES
    Hexa2TetraTopologicalMapping_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Topology.Mapping)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
