cmake_minimum_required(VERSION 3.12)

project(Sofa.GL.Component.Engine_test)

set(SOURCE_FILES
    TextureInterpolation_DataUpdate.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.GL.Component)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
