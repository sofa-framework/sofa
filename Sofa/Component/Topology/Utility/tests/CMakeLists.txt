cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Topology.Utility_test)

set(SOURCE_FILES 
    TopologicalChangeProcessor_test.cpp
    TopologyChecker_test.cpp
)

add_definitions("-DSOFA_COMPONENT_TOPOLOGY_UTILITY_TEST_SCENES_DIR=\"${CMAKE_SOURCE_DIR}/examples/Component/Topology/Container/Dynamic\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Utility)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
