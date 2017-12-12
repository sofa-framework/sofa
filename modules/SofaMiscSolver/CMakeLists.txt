cmake_minimum_required(VERSION 3.1)
project(SofaMiscSolver)

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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
