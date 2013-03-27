cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Tutorials")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

#add_subdirectory("chainHybrid")
#add_subdirectory("compositeObjects")
#add_subdirectory("houseOfCards")
#add_subdirectory("mixedPendulum")
#add_subdirectory("objectCreator")
add_subdirectory("oneParticle")
#add_subdirectory("oneParticleWithSofaTypedefs")
#add_subdirectory("oneTetrahedron")

include(${SOFA_CMAKE_DIR}/post.cmake)
