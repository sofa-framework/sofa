cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaSimulation")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaSimulation_ACTUAL_SOURCE_DIR ${SofaSimulation_SOURCE_DIR}/../modules/sofa/simulation)


set(SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS)
## SofaSimulation dependencies
##
find_package(SofaFramework REQUIRED)

list(APPEND SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS ${SofaFramework_INCLUDE_DIRS})
#include_directories(${SofaFramework_INCLUDE_DIRS})

configure_file(config.h.in
               sofa/simulation/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/simulation/config.h
        DESTINATION include/sofa/simulation)
# make sure everyone in the build tree can see <sofa/simulation/config.h>
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SOFASIMULATION_VERSION ${SofaFramework_VERSION})

add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/common ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCommon)
add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/tree ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationTree)
add_subdirectory(${SofaSimulation_ACTUAL_SOURCE_DIR}/graph ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationGraph)

# IDE folder
set_target_properties(SofaSimulationCommon PROPERTIES FOLDER "SofaSimulation")
set_target_properties(SofaSimulationGraph PROPERTIES FOLDER "SofaSimulation")
set_target_properties(SofaSimulationTree PROPERTIES FOLDER "SofaSimulation")

include(CMakePackageConfigHelpers)

# SofaSimulationConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(SofaSimulationConfigVersion.cmake
                                 VERSION ${SofaFramework_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationConfigVersion.cmake
        DESTINATION lib/cmake/SofaSimulation)

# SofaSimulationConfig.cmake (build tree)
set(SOFASIMULATION_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaSimulationConfig.cmake.in
                              SofaSimulationConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFASIMULATION_INCLUDE_DIR)

# SofaSimulationConfig.cmake (installed)
set(SOFASIMULATION_INCLUDE_DIR include)
configure_package_config_file(SofaSimulationConfig.cmake.in
                              SofaSimulationConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaSimulation
                              PATH_VARS SOFASIMULATION_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationConfigInstall.cmake
        DESTINATION lib/cmake/SofaSimulation
        RENAME SofaSimulationConfig.cmake)

# SofaSimulationTargets.cmake (installed)
install(EXPORT SofaSimulationTargets
        DESTINATION lib/cmake/SofaSimulation)
