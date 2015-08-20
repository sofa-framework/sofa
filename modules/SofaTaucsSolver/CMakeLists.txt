include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
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

AddCompilerDefinitions("SOFA_BUILD_TAUCS_SOLVER")

#if(WIN32 OR SOFA-EXTERNAL_TAUCS_MT)
#	AddLinkerDependencies(metis)
#endif()

if(WIN32)
	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")

	AddLinkerDependencies(blas lapack)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
