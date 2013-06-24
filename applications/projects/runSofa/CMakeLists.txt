include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

	Main.cpp
	)

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
endif()

if(APPLE)
    set(RC_FILES "../Modeler.icns")
else()
    set(RC_FILES "sofa.rc")
endif()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(SofaGuiMain SofaComponentMain)

if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

