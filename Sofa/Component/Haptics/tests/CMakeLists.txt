cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Haptics_test)

set(SOURCE_FILES
    LCPForceFeedback_test.cpp
)

add_definitions("-DSOFA_COMPONENT_HAPTICS_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Haptics Sofa.Component.StateContainer)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
