cmake_minimum_required(VERSION 3.12)
project(Sofa.Simulation LANGUAGES CXX)

set(SOFASIMULATION_SOURCE_DIR "src/sofa/simulation")

set(SOFASIMULATION_DIRS Core Common Graph)
set(SOFASIMULATION_TARGETS)
foreach(simulation_dir ${SOFASIMULATION_DIRS})
    sofa_add_subdirectory(library ${simulation_dir} ${PROJECT_NAME}.${simulation_dir} ON)
    if(TARGET ${PROJECT_NAME}.${simulation_dir})
        list(APPEND SOFASIMULATION_TARGETS ${PROJECT_NAME}.${simulation_dir})
    endif()
endforeach()

set(HEADER_FILES
    ${SOFASIMULATION_SOURCE_DIR}/config.h.in
    ${SOFASIMULATION_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFASIMULATION_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFASIMULATION_TARGETS})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_SIMULATION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_SIMULATION_BUILD_TESTS)
    add_subdirectory(simutest)
endif()
