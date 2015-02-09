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

# Configuration file for runSofa
set(SOFA_SHARE_DIR ${SOFA_SRC_DIR}/share)
set(SOFA_EXAMPLES_DIR ${SOFA_SRC_DIR}/examples)
configure_file(runSofa-config.xml.in ${CMAKE_BINARY_DIR}/etc/runSofa-config.xml)

if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
    AddCompilerDefinitions("SOFA_HAVE_DAG" )
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

