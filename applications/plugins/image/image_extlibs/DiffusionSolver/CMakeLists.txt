include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(HEADER_FILES

    DiffusionSolver.h
	)
	
set(SOURCE_FILES

	DiffusionSolver.cpp
	)

	
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})



find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_C_LINK_FLAGS_RELEASE "${CMAKE_C_LINK_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
else()
    # parallelism is essential for this solver, force to compile with g++ if the current compiler does not support openmp
    # if you computer does not have g++ I am not sure what to do...
    message( "WARNING: DiffusionSolver: your compiler does not implement openmp, trying to compile with g++ and openmp" )
    set (CMAKE_CXX_COMPILER             "/usr/bin/g++")
    set ( CMAKE_CXX_COMPILER_ARG1 "" )
    set (CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    set (CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} -fopenmp")
endif()

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


include(${SOFA_CMAKE_DIR}/postProject.cmake)
