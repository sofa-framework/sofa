include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMisc.h 
    # ../misc/MeshTetraStuffing.h

    )
    
set(SOURCE_FILES

    ../initMisc.cpp 
    # ../misc/MeshTetraStuffing.cpp
  
    )

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "../linearsolver/ParallelCGLinearSolver.h")
	list(APPEND HEADER_FILES "../linearsolver/ParallelCGLinearSolver.inl")

	list(APPEND SOURCE_FILES "../linearsolver/ParallelCGLinearSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC" )
AddLinkerDependencies(SofaMeshCollision SofaMiscTopology SofaTopologyMapping )
        
include(${SOFA_CMAKE_DIR}/postProject.cmake)
