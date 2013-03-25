cmake_minimum_required(VERSION 2.8)

project("SofaPardisoSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

if(SOFA_HAVE_PARDISO)
set(HEADER_FILES

    SparsePARDISOSolver.h

    )
set(SOURCE_FILES

    SparsePARDISOSolver.cpp

    )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_PARDISO_SOLVER")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
