cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/pre.cmake)

add_subdirectory(common)
add_subdirectory(tree)
if(SIMULATION_GRAPH_DAG)
	add_subdirectory(graph)
endif()
if(SIMULATION_GRAPH_BGL)
	add_subdirectory(bgl)
endif()

include(${SOFA_CMAKE_DIR}/post.cmake)
