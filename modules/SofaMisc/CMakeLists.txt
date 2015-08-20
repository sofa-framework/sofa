include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initMisc.h
    MeshTetraStuffing.h

    )

set(SOURCE_FILES

    initMisc.cpp
    MeshTetraStuffing.cpp

    )

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "ParallelCGLinearSolver.h")
	list(APPEND HEADER_FILES "ParallelCGLinearSolver.inl")

	list(APPEND SOURCE_FILES "ParallelCGLinearSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC" )
AddLinkerDependencies(SofaMeshCollision SofaMiscTopology SofaTopologyMapping )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
