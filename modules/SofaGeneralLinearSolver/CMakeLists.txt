cmake_minimum_required(VERSION 3.1)
project(SofaGeneralLinearSolver)

set(HEADER_FILES
    config.h
    initGeneralLinearSolver.h
    )

set(SOURCE_FILES
    initGeneralLinearSolver.cpp
    )

list(APPEND HEADER_FILES
    BTDLinearSolver.h
    BTDLinearSolver.inl
    CholeskySolver.h
    CholeskySolver.inl
    MinResLinearSolver.h
    MinResLinearSolver.inl
    )
list(APPEND SOURCE_FILES
    BTDLinearSolver.cpp
    CholeskySolver.cpp
    MinResLinearSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaBaseLinearSolver)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_LINEAR_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
