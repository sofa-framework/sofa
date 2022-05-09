cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Mass_test)

set(SOURCE_FILES
    DiagonalMass_test.cpp
    MeshMatrixMass_test.cpp
    UniformMass_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Mass Sofa.Component.Topology.Container.Dynamic Sofa.Component.StateContainer)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
