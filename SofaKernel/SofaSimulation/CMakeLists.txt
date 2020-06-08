cmake_minimum_required(VERSION 3.12)
project(SofaSimulation)

set(SOFASIMULATION_TARGETS
    SofaSimulationCommon
    SofaSimulationGraph
    SofaSimulationTree
    )

foreach(TARGET ${SOFASIMULATION_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
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
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFASIMULATION_TARGETS})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
