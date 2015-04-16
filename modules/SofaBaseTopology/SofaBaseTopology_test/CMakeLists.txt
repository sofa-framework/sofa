cmake_minimum_required(VERSION 2.8)

project(SofaBaseTopology_test)

set(SOURCE_FILES
    BezierTetrahedronTopology_test.cpp
    TetrahedronNumericalIntegration_test.cpp)

add_definitions("-DSOFABASETOPOLOGY_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
