cmake_minimum_required(VERSION 2.8)

project(SofaLoader_test)

set(SOURCE_FILES
    MeshVTKLoader_test.cpp
    MeshObjLoader_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
