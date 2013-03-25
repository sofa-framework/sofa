cmake_minimum_required(VERSION 2.8)

project("SofaDenseSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initDenseSolver.h 
    linearsolver/LULinearSolver.h 
    linearsolver/NewMatVector.h 
    linearsolver/NewMatMatrix.h
    
    )
    
set(SOURCE_FILES

    initDenseSolver.cpp 
    linearsolver/LULinearSolver.cpp 
    linearsolver/NewMatCGLinearSolver.cpp 
    linearsolver/NewMatCholeskySolver.cpp
    
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseLinearSolver )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_DENSE_SOLVER")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
