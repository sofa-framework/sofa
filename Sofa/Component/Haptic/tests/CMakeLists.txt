cmake_minimum_required(VERSION 3.1)

project(Sofa.Component.Haptic_test)

set(SOURCE_FILES
    LCPForceFeedback_test.cpp
)

add_definitions("-DSOFA_COMPONENT_HAPTIC_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Haptic Sofa.Component.StateContainer)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
