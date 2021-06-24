cmake_minimum_required(VERSION 3.12)

project(SofaGeneralExplicitOdeSolver_test)

set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    CentralDifferenceExplicitSolverDynamic_test.cpp
    RungeKutta2ExplicitSolverDynamic_test.cpp
    RungeKutta4ExplicitSolverDynamic_test.cpp
    )

sofa_find_package(SofaGeneralExplicitOdeSolver REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaGeneralExplicitOdeSolver SofaBoundaryCondition SofaDeformable SceneCreator)

get_target_property(SOFAEXPLICITODESOLVERTEST_INCLUDES SofaExplicitOdeSolver_test INCLUDE_DIRECTORIES)

# to use MassSpringSystemCreation
target_include_directories(${PROJECT_NAME} PUBLIC ${SOFAEXPLICITODESOLVERTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
