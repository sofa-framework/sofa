set(SOLUTION_NAME "SofaSimulation")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaSimulationCommon" PATH "${CMAKE_CURRENT_SOURCE_DIR}/common")
RegisterDependencies("SofaSimulationTree" PATH "${CMAKE_CURRENT_SOURCE_DIR}/tree")
RegisterDependencies("SofaSimulationGraph" OPTION "SOFA-LIB_SIMULATION_GRAPH_DAG" COMPILE_DEFINITIONS "SOFA_HAVE_DAG" PATH "${CMAKE_CURRENT_SOURCE_DIR}/graph")

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
