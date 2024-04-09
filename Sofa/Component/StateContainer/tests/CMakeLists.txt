cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.StateContainer_test)

set(SOURCE_FILES
    MechanicalObject_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.StateContainer)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
