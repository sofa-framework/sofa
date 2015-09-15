cmake_minimum_required(VERSION 2.8.12)
project(SofaSimulation)

## Dependencies
find_package(SofaFramework REQUIRED)

## Version
set(SOFASIMULATION_VERSION ${SofaFramework_VERSION})

## sofa/simulation/config.h
configure_file(config.h.in sofa/simulation/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/simulation/config.h DESTINATION include/sofa/simulation)

## Installation macro for subdirectories
macro(SofaSimulation_install lib)
    install(TARGETS ${lib}
            EXPORT SofaSimulationTargets
            RUNTIME DESTINATION bin COMPONENT SofaSimulation_libraries
            LIBRARY DESTINATION lib COMPONENT SofaSimulation_libraries
            ARCHIVE DESTINATION lib COMPONENT SofaSimulation_libraries
            PUBLIC_HEADER DESTINATION include/${lib} COMPONENT SofaSimulation_headers)
endmacro()

## Subdirectories
add_subdirectory(../modules/sofa/simulation/common ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCommon)
add_subdirectory(../modules/sofa/simulation/tree ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationTree)
add_subdirectory(../modules/sofa/simulation/graph ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationGraph)

set(SOFASIMULATION_TARGETS
    SofaSimulationCommon
    SofaSimulationGraph
    SofaSimulationTree
)

foreach(TARGET ${SOFASIMULATION_TARGETS})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") # For sofa/simulation/config.h
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaSimulation) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFASIMULATION_VERSION})
endforeach()

## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaSimulation ${SOFASIMULATION_VERSION} "${CMAKE_CURRENT_SOURCE_DIR}/../modules" "include")
