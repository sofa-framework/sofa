cmake_minimum_required(VERSION 3.1)
project(SofaSimulationHelperTesting)

find_package(SofaHelperTesting REQUIRED)
find_package(SofaSimulation REQUIRED)

set(HEADER_FILES
    src/sofa/simulation/helper/testing/config.h
    src/sofa/simulation/helper/testing/BaseSimulationTest.h
)

set(SOURCE_FILES
    src/sofa/simulation/helper/testing/BaseSimulationTest.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_compile_options(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFASIMULATIONHELPERTESTING" )

## Express the dependency to load the link and include information. With the PUBLIC keyword those
## are then made available to third party using this package
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelperTesting SofaSimulationGraph)

## Add to the list of include directory the one with the current files.
## In the src directory is a mimickin of the namespace structure so we always know where
## files are.
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

sofa_create_package(${PROJECT_NAME} 1.0 ${PROJECT_NAME} src)

add_subdirectory(SofaGTestMain)
