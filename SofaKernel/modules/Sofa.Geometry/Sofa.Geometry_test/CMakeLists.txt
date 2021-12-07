cmake_minimum_required(VERSION 3.12)

project(Sofa.Geometry_test)

set(SOURCE_FILES
    Edge_test.cpp
    Triangle_test.cpp
    Quad_test.cpp
    Tetrahedron_test.cpp
    Hexahedron_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Geometry)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
