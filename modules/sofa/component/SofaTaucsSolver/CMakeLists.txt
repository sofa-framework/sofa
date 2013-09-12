include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initTaucsSolver.h 
    ../linearsolver/SparseTAUCSSolver.h 
    ../linearsolver/IncompleteTAUCSSolver.h 
    )
    
set(SOURCE_FILES

	../initTaucsSolver.cpp
    ../linearsolver/SparseTAUCSSolver.cpp 
    ../linearsolver/IncompleteTAUCSSolver.cpp 
    )
    
if(SOFA-EXTERNAL_TAUCS_MT)
	list(APPEND HEADER_FILES "linearsolver/SparseTAUCSLLtSolver.h")

	list(APPEND SOURCE_FILES "linearsolver/SparseTAUCSLLtSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_TAUCS_SOLVER")

if(WIN32 OR SOFA-EXTERNAL_TAUCS_MT)
	AddLinkerDependencies(metis)
endif()

if(WIN32)
	AddLinkerDependencies(blas_win32_MT lapack_win32_MT)
endif()
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
