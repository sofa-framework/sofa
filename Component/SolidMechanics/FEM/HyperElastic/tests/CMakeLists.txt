cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.SolidMechanics.FEM.HyperElastic_test)

set(SOURCE_FILES
    TetrahedronHyperelasticityFEMForceField_params_test.cpp
    TetrahedronHyperelasticityFEMForceField_scene_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SolidMechanics.FEM.HyperElastic)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
