cmake_minimum_required(VERSION 3.12)

project(SofaDeformable_test)

set(SOURCE_FILES
    RestShapeSpringsForceField_test.cpp
    StiffSpringForceField_test.cpp
    PolynomialRestShapeSpringsForceField_test.cpp
    )

sofa_find_package(SofaDeformable REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaBaseMechanics SofaDeformable SofaImplicitOdeSolver SceneCreator)

get_target_property(SOFASIMPLEFEMTEST_INCLUDES SofaSimpleFem_test INCLUDE_DIRECTORIES)

# to use ForceFieldTestCreation
target_include_directories(${PROJECT_NAME} PUBLIC ${SOFASIMPLEFEMTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
