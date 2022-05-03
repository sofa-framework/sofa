cmake_minimum_required(VERSION 3.12)

project(Sofa.Simulation LANGUAGES CXX)

add_subdirectory(Core)
add_subdirectory(Common)
add_subdirectory(Graph)

set(SOFASIMULATION_SOURCE_DIR "src/sofa/simulation")

set(SOFASIMULATION_MODULES Sofa.Simulation.Core Sofa.Simulation.Common Sofa.Simulation.Graph)
foreach(simulation_module ${SOFASIMULATION_MODULES})
    sofa_find_package(${simulation_module} REQUIRED)
endforeach()

set(HEADER_FILES
    ${SOFASIMULATION_SOURCE_DIR}/config.h.in
    ${SOFASIMULATION_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFASIMULATION_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFASIMULATION_MODULES})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
