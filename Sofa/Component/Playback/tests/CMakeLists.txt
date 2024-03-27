cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Playback_test)

set(SOURCE_FILES
    ReadState_test.cpp
    WriteState_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_PLAYBACK_TEST_FILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\"")
add_definitions("-DSOFA_COMPONENT_PLAYBACK_TEST_BUILD_DIR=\"${CMAKE_BINARY_DIR}/\"")

target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Playback Sofa.Component.StateContainer Sofa.Component.ODESolver.Backward Sofa.Component.LinearSolver.Iterative Sofa.Component.Mass)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
