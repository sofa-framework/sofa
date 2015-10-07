cmake_minimum_required(VERSION 2.8.12)
project(SofaImplicitOdeSolver)

set(HEADER_FILES
    EulerImplicitSolver.h
    StaticSolver.h
    VariationalSymplecticSolver.h
    config.h
    initImplicitODESolver.h
)

set(SOURCE_FILES
    EulerImplicitSolver.cpp
    StaticSolver.cpp
    VariationalSymplecticSolver.cpp
    initImplicitODESolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_IMPLICIT_ODE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
