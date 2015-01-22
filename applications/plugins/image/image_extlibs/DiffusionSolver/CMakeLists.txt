include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(HEADER_FILES

    DiffusionSolver.h
	)
	
set(SOURCE_FILES

	DiffusionSolver.cpp
	)

	
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})



if( CMAKE_BUILD_TYPE STREQUAL "Release" )

    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set (CMAKE_C_LINK_FLAGS_RELEASE "${CMAKE_C_LINK_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        addCompilerDefinitions("USING_OMP_PRAGMAS")
    else()
#       if("${CMAKE_SYSTEM}" MATCHES "Linux")
        if(UNIX AND NOT APPLE)
            # parallelism is essential for this solver, force to compile with g++ if the current compiler does not support openmp
            # if your computer does not have g++ I am not sure what to do...
            message( "WARNING: DiffusionSolver: your compiler does not implement openmp, trying to compile with g++ and openmp" )
            set (CMAKE_CXX_COMPILER   "/usr/bin/g++")
            set (CMAKE_CXX_COMPILER_ARG1 "")
            set (CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
            set (CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} -fopenmp")
            addCompilerDefinitions("USING_OMP_PRAGMAS")
        else()
            # no openmp on mac?
            message( "WARNING: DiffusionSolver: your compiler does not implement openmp, the solver will be single-threaded and very inefficient..." )
        endif()
    endif()

    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

endif()


include(${SOFA_CMAKE_DIR}/postProject.cmake)
