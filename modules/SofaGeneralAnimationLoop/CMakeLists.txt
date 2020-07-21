cmake_minimum_required(VERSION 3.12)
project(SofaGeneralAnimationLoop)

set(HEADER_FILES
    config.h
    initGeneralAnimationLoop.h
    )

set(SOURCE_FILES
    initGeneralAnimationLoop.cpp
    )

list(APPEND HEADER_FILES
    MultiStepAnimationLoop.h
    MultiTagAnimationLoop.h
    MechanicalMatrixMapper.h
    MechanicalMatrixMapper.inl)

list(APPEND SOURCE_FILES
    MultiStepAnimationLoop.cpp
    MultiTagAnimationLoop.cpp
    MechanicalMatrixMapper.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaBaseLinearSolver)

sofa_install_targets(SofaGeneral ${PROJECT_NAME} "SofaGeneral/${PROJECT_NAME}")
