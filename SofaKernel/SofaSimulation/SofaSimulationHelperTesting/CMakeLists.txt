cmake_minimum_required(VERSION 3.1)
project(SofaSimulationHelperTesting)

find_package(SofaHelperTesting REQUIRED)
find_package(SofaSimulation REQUIRED)

set(HEADER_FILES
    src/sofa/simulation/helper/testing/BaseSimulationTest.h
)

set(SOURCE_FILES
    src/sofa/simulation/helper/testing/BaseSimulationTest.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

## Express the dependency to load the link and include information. With the PUBLIC keyword those
## are then made available to third party using this package
target_link_libraries(SofaSimulationHelperTesting PUBLIC SofaHelperTesting SofaSimulationGraph)

## Add to the list of include directory the one with the current files.
## In the src directory is a mimickin of the namespace structure so we always know where
## files are.
target_include_directories(SofaSimulationHelperTesting PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

sofa_create_package(SofaSimulationHelperTesting 1.0 SofaSimulationHelperTesting src)

add_subdirectory(SofaGTestMain)
