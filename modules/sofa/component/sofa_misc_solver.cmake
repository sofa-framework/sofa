cmake_minimum_required(VERSION 2.8)

project("SofaMiscSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initMiscSolver.h 
    odesolver/DampVelocitySolver.h 
    odesolver/NewmarkImplicitSolver.h

    )
    
set(SOURCE_FILES

    initMiscSolver.cpp 
    odesolver/DampVelocitySolver.cpp 
    odesolver/NewmarkImplicitSolver.cpp 
    
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_MISC_SOLVER")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
