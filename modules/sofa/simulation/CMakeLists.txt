cmake_minimum_required(VERSION 2.8)
set(SOLUTION_NAME "SofaSimulation")
project(${SOLUTION_NAME})

message("Install prefix: ${CMAKE_INSTALL_PREFIX}")


##
## SofaSimulation version 
##
set( SOFASIMULATION_MAJOR_VERSION 0 )
set( SOFASIMULATION_MINOR_VERSION 0 )
set( SOFASIMULATION_BUILD_VERSION 1 )

set( SOFASIMULATION_VERSION ${SOFASIMULATION_MAJOR_VERSION}.${SOFASIMULATION_MINOR_VERSION}.${SOFASIMULATION_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.") 

set(SOFASIMULATION_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR})
set(SOFASIMULATION_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR})
set(SOFASIMULATION_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOFASIMULATION_BUILD_CMAKE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulation)


message("Build dir : ${CMAKE_CURRENT_BINARY_DIR}")

# Offer the user the choice of overriding the installation directories
set(SOFASIMULATION_INSTALL_LIB_DIR     lib     					  CACHE PATH "Installation directory for libraries")
set(SOFASIMULATION_INSTALL_BIN_DIR     bin     					  CACHE PATH "Installation directory for executables")
set(SOFASIMULATION_INSTALL_INCLUDE_DIR include 					  CACHE PATH "Installation directory for header files")
set(SOFASIMULATION_INSTALL_CMAKE_DIR   lib/cmake/SofaSimulation   CACHE PATH "Installation directory for CMake files")

## SofaSimulation dependencies  
##
find_package(SofaFramework REQUIRED)
include_directories( ${SofaFramework_INCLUDE_DIRS} )

configure_file( ${SofaSimulation_SOURCE_DIR}/SofaSimulation.config.in 
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaSimulation.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaSimulation.h 
         DESTINATION "${SOFASIMULATION_INSTALL_INCLUDE_DIR}/sofa")
# make sure everyone in the build tree can see <sofa/SofaSimulation.h>
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${SofaSimulation_SOURCE_DIR}/../../)

add_subdirectory(${SofaSimulation_SOURCE_DIR}/common ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCommon )
add_subdirectory(${SofaSimulation_SOURCE_DIR}/tree   ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationTree   )
add_subdirectory(${SofaSimulation_SOURCE_DIR}/graph  ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationGraph  ) 

message( "SofaFramework_TARGETS : ${SofaFramework_TARGETS}")
list(APPEND SOFASIMULATION_TARGETS SofaSimulationCommon SofaSimulationTree SofaSimulationGraph ${SofaFramework_TARGETS} )

include(CMakePackageConfigHelpers)

# SofaSimulationConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationConfigVersion.cmake
                                 VERSION ${SOFASIMULATION_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationConfigVersion.cmake
        DESTINATION ${SOFASIMULATION_INSTALL_CMAKE_DIR})
                                                          

# SofaSimulationConfig.cmake (build tree)
set(SOFASIMULATION_INCLUDE_DIR "${SOFASIMULATION_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaSimulationConfig.cmake.in 
                              ${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFASIMULATION_INCLUDE_DIR)

# SofaSimulationConfig.cmake (installed)
set(SOFASIMULATION_INCLUDE_DIR "${SOFASIMULATION_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaSimulationConfig.cmake.in 
                              ${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFASIMULATION_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFASIMULATION_INCLUDE_DIR )

install(FILES ${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationConfigInstall.cmake
        DESTINATION ${SOFASIMULATION_INSTALL_CMAKE_DIR}
        RENAME SofaSimulationConfig.cmake)

# SofaSimulationTargets.cmake (build tree)        
export(TARGETS ${SOFASIMULATION_TARGETS}
       FILE ${SOFASIMULATION_BUILD_CMAKE_DIR}/SofaSimulationTargets.cmake )                                        
                                                          
# SofaSimulationTargets.cmake (installed)
install(EXPORT SofaSimulationTargets
        DESTINATION ${SOFASIMULATION_INSTALL_CMAKE_DIR} )


# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE SofaSimulation)

#include(SofaSimulationCPack.cmake)
