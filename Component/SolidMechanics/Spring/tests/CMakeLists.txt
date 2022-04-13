cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.SolidMechanics.Spring_test)

set(SOURCE_FILES
    PolynomialRestShapeSpringsForceField_test.cpp
    RestShapeSpringsForceField_test.cpp
    StiffSpringForceField_test.cpp
    TriangularBendingSprings_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SolidMechanics.Spring)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
