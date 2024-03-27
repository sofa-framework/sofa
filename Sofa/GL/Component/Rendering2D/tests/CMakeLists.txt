cmake_minimum_required(VERSION 3.22)

project(Sofa.GL.Component.Rendering2D_test)

set(SOURCE_FILES
    OglLabel_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.GL.Component)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
