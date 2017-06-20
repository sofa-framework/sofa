cmake_minimum_required(VERSION 3.1)

project(SofaBaseTopology_test)

set(SOURCE_FILES
    PointSetTopology_test.cpp
    RegularGridTopology_test.cpp
    TetrahedronNumericalIntegration_test.cpp
    TriangleNumericalIntegration_test.cpp)

add_definitions("-DSOFABASETOPOLOGY_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
