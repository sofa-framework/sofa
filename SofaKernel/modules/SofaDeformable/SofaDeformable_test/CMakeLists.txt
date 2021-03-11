cmake_minimum_required(VERSION 3.12)

project(SofaDeformable_test)

set(SOURCE_FILES
    RestShapeSpringsForceField_test.cpp
    StiffSpringForceField_test.cpp
    PolynomialRestShapeSpringsForceField_test.cpp
    )

sofa_find_package(SofaDeformable REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaTest SofaDeformable)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
