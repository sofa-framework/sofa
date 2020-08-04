cmake_minimum_required(VERSION 3.12)
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

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
