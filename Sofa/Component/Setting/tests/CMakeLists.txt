cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Setting_test)

set(SOURCE_FILES
    PrintLog_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_SETTING_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Setting)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
