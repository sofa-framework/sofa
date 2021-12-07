cmake_minimum_required(VERSION 3.12)

project(SofaLoader_test)

set(SOURCE_FILES
    MeshVTKLoader_test.cpp
    MeshOBJLoader_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaLoader)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
