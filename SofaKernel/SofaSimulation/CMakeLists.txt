cmake_minimum_required(VERSION 3.12)
project(SofaSimulation)

set(SOFASIMULATION_MODULES
    SofaSimulationCommon
    SofaSimulationGraph
    )

foreach(MODULE ${SOFASIMULATION_MODULES})
    add_subdirectory(../modules/${MODULE} ${CMAKE_CURRENT_BINARY_DIR}/${MODULE})
    set_target_properties(${MODULE} PROPERTIES FOLDER SofaSimulation)
endforeach()

set(SRC_ROOT src/SofaSimulation)
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaSimulation.h
    )
set(SOURCE_FILES
    ${SRC_ROOT}/initSofaSimulation.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFASIMULATION_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
