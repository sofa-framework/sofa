cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.SolidMechanics_simutest)

set(SOURCE_FILES
    AffinePatch_test.cpp
    LinearElasticity_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Testing SceneCreator)
target_link_libraries(${PROJECT_NAME} Sofa.Component)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
