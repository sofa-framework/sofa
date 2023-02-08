cmake_minimum_required(VERSION 3.12)

project(Sofa.Simulation.Common_test)

set(SOURCE_FILES
    LoadScene_test.cpp
)

add_definitions("-DSOFASIMULATION_TEST_SCENES_DIR=\"${CMAKE_SOURCE_DIR}/examples/Component/Constraint/Projective\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Helper)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

