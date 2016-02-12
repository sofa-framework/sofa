cmake_minimum_required(VERSION 2.8.12)
project(SofaSimulation)

## Dependencies
find_package(SofaFramework REQUIRED)

## Version
set(SOFASIMULATION_VERSION ${SofaFramework_VERSION})

## sofa/simulation/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/sofa/simulation/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/simulation/config.h" DESTINATION include/sofa/simulation)

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
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>") # For sofa/simulation/config.h
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaSimulation) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION ${SOFASIMULATION_VERSION})
endforeach()

## CMake package configurations files
sofa_write_package_config_files(SofaSimulation ${SOFASIMULATION_VERSION})

#
# python wrapping using swig
#
if(SWIG_FOUND AND PYTHONLIBS_FOUND)
    message(STATUS "Add sofa.simulation python package generation using SWIG")
    include_directories(${PYTHON_INCLUDE_PATH})
    include(${SWIG_USE_FILE})

    set_source_files_properties(../modules/sofa/simulation/simulation.i PROPERTIES CPLUSPLUS ON)
    set_source_files_properties(../modules/sofa/simulation/simulation.i PROPERTIES SWIG_FLAGS "-includeall")
    swig_add_module(simulation python ../modules/sofa/simulation/simulation.i)
    swig_link_libraries(simulation ${PYTHON_LIBRARIES} SofaSimulationCommon)
    set_target_properties(${SWIG_MODULE_simulation_REAL_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SWIG_OUTDIR})
    install(TARGETS ${SWIG_MODULE_simulation_REAL_NAME} DESTINATION lib/python2.7/site-packages/sofa)
    install(FILES ${CMAKE_SWIG_OUTDIR}/simulation.py DESTINATION lib/python2.7/site-packages/sofa)
endif()
