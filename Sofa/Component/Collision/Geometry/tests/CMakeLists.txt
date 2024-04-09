cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Collision.Geometry_test)

set(SOURCE_FILES
    Sphere_test.cpp
    Triangle_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Collision.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Geometry Sofa.Component.Collision.Detection.Intersection)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
