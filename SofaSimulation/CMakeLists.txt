cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaSimulation")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set( SofaSimulation_ACTUAL_SOURCE_DIR ${SofaSimulation_SOURCE_DIR}/../modules/sofa/simulation)

##
## SofaSimulation version
##
set( SOFASIMULATION_MAJOR_VERSION 0 )
set( SOFASIMULATION_MINOR_VERSION 0 )
set( SOFASIMULATION_BUILD_VERSION 1 )

set( SOFASIMULATION_VERSION ${SOFASIMULATION_MAJOR_VERSION}.${SOFASIMULATION_MINOR_VERSION}.${SOFASIMULATION_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFASIMULATION_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFASIMULATION_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFASIMULATION_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFASIMULATION_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaSimulation)

# Offer the user the choice of overriding the installation directories
set(SOFASIMULATION_INSTALL_LIB_DIR     lib     					  CACHE PATH "Installation directory for libraries")
set(SOFASIMULATION_INSTALL_BIN_DIR     bin     					  CACHE PATH "Installation directory for executables")
set(SOFASIMULATION_INSTALL_INCLUDE_DIR include 					  CACHE PATH "Installation directory for header files")
set(SOFASIMULATION_INSTALL_CMAKE_DIR   lib/cmake/SofaSimulation   CACHE PATH "Installation directory for CMake files")

set(SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS )
## SofaSimulation dependencies
##
find_package(SofaFramework REQUIRED)

list(APPEND SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS ${SofaFramework_INCLUDE_DIRS})
#include_directories( ${SofaFramework_INCLUDE_DIRS} )

configure_file( ${SofaSimulation_SOURCE_DIR}/SofaSimulation.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaSimulation.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaSimulation.h
         DESTINATION "${SOFASIMULATION_INSTALL_INCLUDE_DIR}/sofa")
# make sure everyone in the build tree can see <sofa/SofaSimulation.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaSimulation_ACTUAL_SOURCE_DIR}/../../)

add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/common ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCommon )
add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/tree   ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationTree   )
add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/graph  ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationGraph  )

# set IDE project filter
if(MSVC)
    set_target_properties(SofaSimulationCommon PROPERTIES FOLDER "SofaSimulation")
    set_target_properties(SofaSimulationTree   PROPERTIES FOLDER "SofaSimulation")
    set_target_properties(SofaSimulationGraph  PROPERTIES FOLDER "SofaSimulation")
endif()

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

# SofaSimulationTargets.cmake (installed)
install(EXPORT SofaSimulationTargets
        DESTINATION ${SOFASIMULATION_INSTALL_CMAKE_DIR} )
