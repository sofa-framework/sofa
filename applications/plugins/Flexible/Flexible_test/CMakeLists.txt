include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    StrainMappings_test.cpp
	)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(Flexible SofaComponentMain SofaSimulationGraph)

if(UNIX)
    AddLinkerDependencies("dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
