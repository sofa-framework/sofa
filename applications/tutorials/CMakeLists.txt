cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Tutorials")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

add_subdirectory("chainHybrid")
add_subdirectory("compositeObject")
add_subdirectory("houseOfCards")
add_subdirectory("mixedPendulum")
add_subdirectory("objectCreator")
add_subdirectory("oneParticle")
#add_subdirectory("oneParticleWithSofaTypedefs") # temporarily discarded since it can not compile with the Flexible plugin
add_subdirectory("oneTetrahedron")
add_subdirectory("anatomyModelling")

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
