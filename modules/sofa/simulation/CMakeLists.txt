cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaSimulation")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaSimulationCommon" PATH "${CMAKE_CURRENT_SOURCE_DIR}/common")
#add_subdirectory(common)

RegisterDependencies("SofaSimulationTree" PATH "${CMAKE_CURRENT_SOURCE_DIR}/tree")
#add_subdirectory(tree)

RegisterDependencies("SofaSimulationGraph" OPTION SIMULATION_GRAPH_DAG COMPILE_DEFINITIONS SOFA_HAVE_DAG PATH "${CMAKE_CURRENT_SOURCE_DIR}/graph")
#if(SIMULATION_GRAPH_DAG)
#	add_subdirectory(graph)
#endif()

RegisterDependencies("SofaSimulationBGL" OPTION SIMULATION_GRAPH_BGL COMPILE_DEFINITIONS SOFA_HAVE_BGL PATH "${CMAKE_CURRENT_SOURCE_DIR}/bgl")
#if(SIMULATION_GRAPH_BGL)
#	add_subdirectory(bgl)
#endif()

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
