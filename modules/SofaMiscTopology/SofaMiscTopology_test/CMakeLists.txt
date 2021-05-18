cmake_minimum_required(VERSION 3.12)

project(SofaMiscTopology_test)


sofa_find_package(SofaMiscTopology REQUIRED)

set(HEADER_FILES)
set(SOURCE_FILES 
    TopologicalChangeProcessor_test.cpp
    TopologyChecker_test.cpp)

add_definitions("-DSOFAMISCTOPOLOGY_TEST_SCENES_DIR=\"${CMAKE_SOURCE_DIR}/examples/Components/topology\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaMiscTopology)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
