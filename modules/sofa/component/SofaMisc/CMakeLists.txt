cmake_minimum_required(VERSION 2.8)

project("SofaMisc")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMisc.h 
    ../misc/MeshTetraStuffing.h

    )
    
set(SOURCE_FILES

    ../initMisc.cpp 
    ../misc/MeshTetraStuffing.cpp
  
    )

if(SOFA_SMP)
	list(APPEND HEADER_FILES "../linearsolver/ParallelCGLinearSolver.h")
	list(APPEND HEADER_FILES "../linearsolver/ParallelCGLinearSolver.inl")

	list(APPEND SOURCE_FILES "../linearsolver/ParallelCGLinearSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MISC" )
set(LINKER_DEPENDENCIES SofaMeshCollision SofaMiscTopology SofaTopologyMapping )
        
include(${SOFA_CMAKE_DIR}/postProject.cmake)
