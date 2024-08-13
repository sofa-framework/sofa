cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Collision.Detection.Intersection_test)

set(SOURCE_FILES
    LocalMinDistance_test.cpp
    MeshNewProximityIntersection_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Collision.Detection.Intersection)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
