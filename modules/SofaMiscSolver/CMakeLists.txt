cmake_minimum_required(VERSION 3.12)
project(SofaMiscSolver)

find_package(SofaSimulation REQUIRED) # SofaSimulationTree

set(HEADER_FILES
    DampVelocitySolver.h
    NewmarkImplicitSolver.h
    config.h
    initMiscSolver.h
)

set(SOURCE_FILES
    DampVelocitySolver.cpp
    NewmarkImplicitSolver.cpp
    initMiscSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_install_targets(SofaMisc ${PROJECT_NAME} "SofaMisc/${PROJECT_NAME}")
