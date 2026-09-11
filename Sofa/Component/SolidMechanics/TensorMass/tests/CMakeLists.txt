cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.SolidMechanics.TensorMass_test)

set(SOURCE_FILES
        TetrahedralTensorMassForceField_test.cpp
        TriangularTensorMassForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.TensorMass)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
