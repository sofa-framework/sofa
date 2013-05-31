include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    compositeObject.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)
if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    AddLinkerDependencies(SofaSimulationGraph)
endif()
if(SOFA-LIB_SIMULATION_GRAPH_BGL)
    AddLinkerDependencies(SofaSimulationBGL)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

