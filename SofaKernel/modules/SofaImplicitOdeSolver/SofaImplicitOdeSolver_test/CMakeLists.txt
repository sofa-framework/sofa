cmake_minimum_required(VERSION 3.12)

project(SofaImplicitOdeSolver_test)

set(SOURCE_FILES
    loadPlugins.cpp
    EulerImplicitSolverStatic_test.cpp
    EulerImplicitSolverDynamic_test.cpp
    SpringSolverDynamic_test.cpp
    StaticSolver_test.cpp)
    
add_definitions("-DSOFAIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaDeformable SofaImplicitOdeSolver SofaBaseMechanics SofaBaseUtils SofaBoundaryCondition SofaEigen2Solver SceneCreator)

get_target_property(SOFAEXPLICITODESOLVERTEST_INCLUDES SofaExplicitOdeSolver_test INCLUDE_DIRECTORIES)

# to use MassSpringSystemCreation
target_include_directories(${PROJECT_NAME} PUBLIC ${SOFAEXPLICITODESOLVERTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
