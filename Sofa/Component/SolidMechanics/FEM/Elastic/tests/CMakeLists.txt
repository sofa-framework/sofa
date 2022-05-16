cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.SolidMechanics.FEM.Elastic_test)

set(SOURCE_FILES
    BeamFEMForceField_test.cpp
    HexahedronFEMForceField_test.cpp
    TetrahedronFEMForceField_test.cpp
    TriangleFEMForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SolidMechanics.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.FEM.Elastic Sofa.Component.StateContainer Sofa.Component.Topology.Container.Dynamic)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
