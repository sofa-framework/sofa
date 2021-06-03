cmake_minimum_required(VERSION 3.12)

project(SofaImplicitOdeSolver_test)

set(HEADER_FILES
    MassSpringSystemCreation.h
)

set(SOURCE_FILES
    loadPlugins.cpp
    EulerImplicitSolverStatic_test.cpp
    EulerImplicitSolverDynamic_test.cpp
    SpringSolverDynamic_test.cpp
    StaticSolver_test.cpp)
    
add_definitions("-DSOFAIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaDeformable SofaImplicitOdeSolver SofaBaseMechanics SofaBaseUtils SofaBoundaryCondition SofaEigen2Solver SceneCreator)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
