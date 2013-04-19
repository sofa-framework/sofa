cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Tutorials")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("Standard_test" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/chainHybrid")
RegisterDependencies("CompositeObject" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/compositeObject")
RegisterDependencies("HouseOfCards" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/houseOfCards")
RegisterDependencies("MixedPendulum" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/mixedPendulum")
RegisterDependencies("ObjectCreator" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/objectCreator")
RegisterDependencies("OneParticle" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/oneParticle")
#RegisterDependencies("OneParticleWithSofaTypedefs" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/oneParticleWithSofaTypedefs")
RegisterDependencies("OneTetrahedron" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/oneTetrahedron")
#RegisterDependencies("anatomyModelling" OPTION OPTION_TUTORIALS PATH "${CMAKE_CURRENT_SOURCE_DIR}/anatomyModelling") # currently there is no CMakeLists.txt

#add_subdirectory("chainHybrid")
#add_subdirectory("compositeObject")
#add_subdirectory("houseOfCards")
#add_subdirectory("mixedPendulum")
#add_subdirectory("objectCreator")
#add_subdirectory("oneParticle")
##add_subdirectory("oneParticleWithSofaTypedefs") # temporarily discarded since it can not compile with the Flexible plugin
#add_subdirectory("oneTetrahedron")
##add_subdirectory("anatomyModelling") # currently there is no CMakeLists.txt

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
