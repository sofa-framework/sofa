cmake_minimum_required(VERSION 3.1)
project(SofaGeneralImplicitOdeSolver)

set(HEADER_FILES
    config.h
    initGeneralImplicitODESolver.h
    )

set(SOURCE_FILES
    initGeneralImplicitODESolver.cpp
    )

list(APPEND HEADER_FILES
    VariationalSymplecticSolver.h
    )
list(APPEND SOURCE_FILES
    VariationalSymplecticSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_IMPLICIT_ODE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
