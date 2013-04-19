cmake_minimum_required(VERSION 2.8)

RegisterDependencies("CenterOfMassMapping" OPTION SIMULATION_GRAPH_BGL PATH "${CMAKE_CURRENT_SOURCE_DIR}/applications/centerOfMassMapping")
RegisterDependencies("CenterOfMassMulti2MappingChain" OPTION SIMULATION_GRAPH_BGL PATH "${CMAKE_CURRENT_SOURCE_DIR}/applications/centerOfMassMulti2MappingChain")
RegisterDependencies("CenterOfMassMultiMappingChain" OPTION SIMULATION_GRAPH_BGL PATH "${CMAKE_CURRENT_SOURCE_DIR}/applications/centerOfMassMultiMappingChain")
RegisterDependencies("SubsetMultiMapping" OPTION SIMULATION_GRAPH_BGL PATH "${CMAKE_CURRENT_SOURCE_DIR}/applications/subsetMultimapping")

#add_subdirectory("applications/centerOfMassMapping")
#add_subdirectory("applications/centerOfMassMulti2MappingChain")
#add_subdirectory("applications/centerOfMassMultiMappingChain")
#add_subdirectory("applications/subsetMultimapping")