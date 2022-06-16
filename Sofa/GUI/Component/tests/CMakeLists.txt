cmake_minimum_required(VERSION 3.12)

project(Sofa.GUI.Component_test)

set(SOURCE_FILES
    ConstraintAttachBodyPerformer_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Component)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
