cmake_minimum_required(VERSION 3.1)
project(SofaSimulation)

## Dependencies
find_package(SofaFramework REQUIRED)

## Version
set(SOFASIMULATION_VERSION ${SofaFramework_VERSION})

## sofa/simulation/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/sofa/simulation/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/simulation/config.h" DESTINATION include/sofa/simulation COMPONENT headers)

## Subdirectories
add_subdirectory(../modules/SofaSimulationCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCommon)
add_subdirectory(../modules/SofaSimulationTree ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationTree)
add_subdirectory(../modules/SofaSimulationGraph ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationGraph)

set(SOFASIMULATION_TARGETS
    SofaSimulationCommon
    SofaSimulationGraph
    SofaSimulationTree
)

foreach(TARGET ${SOFASIMULATION_TARGETS})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>") # For sofa/simulation/config.h
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaSimulation) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION ${SOFASIMULATION_VERSION})
endforeach()

## CMake package configurations files
sofa_write_package_config_files(SofaSimulation ${SOFASIMULATION_VERSION})
