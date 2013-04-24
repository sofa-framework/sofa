cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    compositeObject.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain SofaGuiMain)
if(SIMULATION_GRAPH_DAG)
    list(APPEND LINKER_DEPENDENCIES SofaSimulationGraph)
endif()
if(SIMULATION_GRAPH_BGL)
    list(APPEND LINKER_DEPENDENCIES SofaSimulationBGL)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

