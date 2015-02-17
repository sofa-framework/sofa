include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

	Main.cpp
	)

if(SOFA-LIB_GUI_QT)
	UseQt()
endif()

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(SofaSimulationTree SofaGuiMain SofaComponentMain tinyxml)

if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
    AddCompilerDefinitions("SOFA_HAVE_DAG" )
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
