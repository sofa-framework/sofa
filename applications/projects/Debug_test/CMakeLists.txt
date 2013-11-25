include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    )

set(SOURCE_FILES

	My_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_PROJECT_TEST")
AddLinkerDependencies(SofaGuiMain SofaComponentMain)

if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

