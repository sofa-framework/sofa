cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Diffusion_test)

set(SOURCE_FILES
    TetrahedronDiffusionFEMForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_DIFFUSION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Diffusion Sofa.Component.Mass Sofa.Component.StateContainer Sofa.Component.Topology.Container.Grid)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
