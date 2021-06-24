cmake_minimum_required(VERSION 3.12)

project(SofaGeneralImplicitOdeSolver_test)

set(SOURCE_FILES ""
    )

list(APPEND HEADER_FILES
    )
list(APPEND SOURCE_FILES
    VariationalSymplecticExplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverDynamic_test.cpp
    )

sofa_find_package(SofaGeneralImplicitOdeSolver REQUIRED)

add_definitions("-DSOFAGENERALIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaGeneralImplicitOdeSolver SofaBoundaryCondition SofaDeformable SofaMiscForceField SceneCreator)

get_target_property(SOFAEXPLICITODESOLVERTEST_INCLUDES SofaExplicitOdeSolver_test INCLUDE_DIRECTORIES)

# to use MassSpringSystemCreation
target_include_directories(${PROJECT_NAME} PUBLIC ${SOFAEXPLICITODESOLVERTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
