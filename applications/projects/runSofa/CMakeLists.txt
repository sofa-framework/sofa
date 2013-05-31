include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

	Main.cpp
	)

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
endif()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaGuiMain SofaComponentMain)

if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

