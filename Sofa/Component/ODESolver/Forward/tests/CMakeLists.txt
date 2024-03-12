cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.ODESolver.Forward_test)

set(SOURCE_FILES
    CentralDifferenceExplicitSolverDynamic_test.cpp
    EulerExplicitSolverDynamic_test.cpp
    RungeKutta2ExplicitSolverDynamic_test.cpp
    RungeKutta4ExplicitSolverDynamic_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.ODESolver.Testing Sofa.Component.ODESolver.Forward Sofa.Component.StateContainer)
