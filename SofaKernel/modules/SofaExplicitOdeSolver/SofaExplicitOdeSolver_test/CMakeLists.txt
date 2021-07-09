cmake_minimum_required(VERSION 3.12)

project(SofaExplicitOdeSolver_test)

set(HEADER_FILES
    MassSpringSystemCreation.h
)

set(SOURCE_FILES
    EulerExplicitSolverDynamic_test.cpp)

sofa_find_package(SofaExplicitOdeSolver REQUIRED)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaExplicitOdeSolver SofaBaseMechanics SofaDeformable SofaBoundaryCondition SceneCreator)

# allow use MassSpringSystemCreation to other tests
target_include_directories(${PROJECT_NAME} PUBLIC ..)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
