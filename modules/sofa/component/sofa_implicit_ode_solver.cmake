cmake_minimum_required(VERSION 2.8)

project("SofaImplicitOdeSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initImplicitODESolver.h 
    odesolver/EulerImplicitSolver.h 
    odesolver/StaticSolver.h

    )
    
set(SOURCE_FILES

    initImplicitODESolver.cpp 
    odesolver/EulerImplicitSolver.cpp 
    odesolver/StaticSolver.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_IMPLICIT_ODE_SOLVER")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
