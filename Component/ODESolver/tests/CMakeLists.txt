cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.ODESolver_test)

set(SOURCE_FILES
    EulerExplicitSolverDynamic_test.cpp
    SpringSolverDynamic_test.cpp
    EulerImplicitSolverDynamic_test.cpp              
    StaticSolver_test.cpp
    EulerImplicitSolverStatic_test.cpp               
    VariationalSymplecticExplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverDynamic_test.cpp
    NewmarkImplicitSolverDynamic_test.cpp
    # VariationalSymplecticImplicitSolverNonLinearForceDynamic_test.cpp # disabled, needs a removed component
    RungeKutta2ExplicitSolverDynamic_test.cpp
)

add_definitions("-DSOFACOMPONENTODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Testing.Tools Sofa.Component.ODESolver SofaBaseUtils SofaBaseMechanics SofaDeformable SofaBoundaryCondition SceneCreator)
