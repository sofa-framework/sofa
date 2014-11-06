cmake_minimum_required(VERSION 2.8)
project(SofaSparseSolver)


set(HEADER_FILES

    initSparseSolver.h
    PrecomputedLinearSolver.h
    PrecomputedLinearSolver.inl
    SparseLDLSolverImpl.h
    SparseLDLSolver.h
    SparseLDLSolver.inl
    )

set(SOURCE_FILES

    initSparseSolver.cpp
    PrecomputedLinearSolver.cpp
    SparseLDLSolver.cpp
    )

if( SOFA_HAVE_CSPARSE )
	list(APPEND HEADER_FILES "SparseCholeskySolver.h")
	list(APPEND HEADER_FILES "SparseLUSolver.h")

	list(APPEND SOURCE_FILES "SparseCholeskySolver.cpp")
	list(APPEND SOURCE_FILES "SparseLUSolver.cpp")
 endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaBaseLinearSolver SofaImplicitOdeSolver SofaSimpleFem )
if( SOFA_HAVE_CSPARSE )
target_link_libraries( ${PROJECT_NAME} csparse )
endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFASPARSE_SOLVER_COMPILER_FLAGS "-DSOFA_BUILD_SPARSE_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFASPARSE_SOLVER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSparseSolver_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)
