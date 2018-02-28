cmake_minimum_required(VERSION 3.1)
project(SofaAssimp)

set(ASSIMP_PLUGIN_MAJOR_VERSION 0)
set(ASSIMP_PLUGIN_MINOR_VERSION 1)
set(ASSIMP_PLUGIN_VERSION ${ASSIMP_PLUGIN_MAJOR_VERSION}.${ASSIMP_PLUGIN_MINOR_VERSION})

set(HEADER_FILES
    config.h
    AssimpLoader.h
)

set(SOURCE_FILES
    initAssimpPlugin.cpp
    AssimpLoader.cpp
)

set(README_FILES README_AssimpPlugin.txt)


## Assimp dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

### Dependencies
find_package(Assimp REQUIRED)


## SofaAssimp/config.h
#configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaAssimp/config.h")
#install(FILES "${CMAKE_BINARY_DIR}/include/SofaAssimp/config.h" DESTINATION "include/SofaAssimp")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaHelper)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/..")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${ASSIMP_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAASSIMP")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${ASSIMP_PLUGIN_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

#TODO
if(SOFA_BUILD_TESTS)
   find_package(SofaTest QUIET)
#   add_subdirectory(CGALPlugin_test)
endif()

## Install rules for the library and the headers; CMake package configurations files
#sofa_create_package(SofaAssimp ${ASSIMP_PLUGIN_VERSION} ${PROJECT_NAME} SofaAssimp)

## Install rules for the resources
#install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
