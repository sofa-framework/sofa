cmake_minimum_required(VERSION 2.8.12)
project(SofaTaucsSolver)


set(HEADER_FILES
    IncompleteTAUCSSolver.h
    SparseTAUCSSolver.h
    config.h
    initTaucsSolver.h
)

set(SOURCE_FILES
    IncompleteTAUCSSolver.cpp
    SparseTAUCSSolver.cpp
    initTaucsSolver.cpp
)

# # TODO?
# if(SOFA-EXTERNAL_TAUCS_MT)
#    list(APPEND HEADER_FILES linearsolvSparseTAUCSLLtSolver.h)
#    list(APPEND SOURCE_FILES linearsolvSparseTAUCSLLtSolver.cpp)
# endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseCollision SofaLoader SofaMeshCollision)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TAUCS_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if(WIN32)
    target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../extlibs/SuiteSparse/cholmod/Include>")
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE blas lapack)
endif()

SofaGeneral_install(${PROJECT_NAME})
