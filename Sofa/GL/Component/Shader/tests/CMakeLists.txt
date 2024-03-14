cmake_minimum_required(VERSION 3.12)

project(Sofa.GL.Component.Shader_test)

## MAC-OS crash when we are testing opengl feature.
if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set(SOURCE_FILES
     Light_test.cpp
     LightManager_test.cpp)


    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
    target_link_libraries(${PROJECT_NAME} Sofa.GL.Component)

    add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
endif()
