cmake_minimum_required(VERSION 2.8)
project(SofaTaucsSolver)


set(HEADER_FILES

    initTaucsSolver.h
    SparseTAUCSSolver.h
    IncompleteTAUCSSolver.h
    )

set(SOURCE_FILES

	initTaucsSolver.cpp
    SparseTAUCSSolver.cpp
    IncompleteTAUCSSolver.cpp
    )

#if(SOFA-EXTERNAL_TAUCS_MT)
#	list(APPEND HEADER_FILES "linearsolvSparseTAUCSLLtSolver.h")
#
#	list(APPEND SOURCE_FILES "linearsolvSparseTAUCSLLtSolver.cpp")
#endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaBaseCollision SofaLoader SofaMeshCollision )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

if(WIN32)
    include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
    target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include>")
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
    target_link_libraries( ${PROJECT_NAME} blas lapack )
endif()

set( SOFATAUCS_SOLVER_COMPILER_FLAGS "-DSOFA_BUILD_TAUCS_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFATAUCS_SOLVER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaTaucsSolver_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)
