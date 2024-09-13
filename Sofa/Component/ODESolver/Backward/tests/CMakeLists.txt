cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.ODESolver.Backward_test)

set(SOURCE_FILES
    EulerImplicitSolverDynamic_test.cpp
    EulerImplicitSolverStatic_test.cpp
    EulerImplicitSolver_withDamping_test.cpp
    NewmarkImplicitSolverDynamic_test.cpp
    StaticSolver_test.cpp
    SpringSolverDynamic_test.cpp
    VariationalSymplecticExplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverDynamic_test.cpp
)

add_definitions("-DSOFACOMPONENTODESOLVERBACKWARD_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.ODESolver.Testing Sofa.Component.ODESolver.Backward Sofa.Component.StateContainer)
