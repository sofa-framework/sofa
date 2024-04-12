cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.SolidMechanics.FEM.HyperElastic_test)

set(SOURCE_FILES
    Material_test.cpp
    TetrahedronHyperelasticityFEMForceField_params_test.cpp
    TetrahedronHyperelasticityFEMForceField_scene_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME}
    Sofa.Component.SolidMechanics.FEM.HyperElastic
    Sofa.Component.SolidMechanics.Testing
    Sofa.Component.StateContainer
    Sofa.Component.ODESolver.Backward)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
