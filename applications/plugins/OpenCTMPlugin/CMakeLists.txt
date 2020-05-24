cmake_minimum_required(VERSION 3.1)
project(OpenCTMPlugin VERSION 0.1)

find_package(SofaBase REQUIRED)
find_package(OpenCTM REQUIRED)

set(SOfA_OPENCTM_SRC_DIR src/SofaOpenCTM)

set(HEADER_FILES
    ${SOfA_OPENCTM_SRC_DIR}/config.h.in
    ${SOfA_OPENCTM_SRC_DIR}/OpenCTMExporter.h
    ${SOfA_OPENCTM_SRC_DIR}/OpenCTMLoader.h
)

set(SOURCE_FILES
    ${SOfA_OPENCTM_SRC_DIR}/OpenCTMExporter.cpp
    ${SOfA_OPENCTM_SRC_DIR}/OpenCTMLoader.cpp
    ${SOfA_OPENCTM_SRC_DIR}/initOpenCTMPlugin.cpp
)

set(README_FILES OpenCTMPlugin.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Set define dllimport/dllexport mechanism on Windows.
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFAOPENCTMPLUGIN")

target_link_libraries(${PROJECT_NAME} SofaBaseTopology SofaOpenglVisual ${OPENCTM_LIBRARY})

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${OPENCTM_INCLUDE_PATH}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")


## Install rules for the library; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR ${SOfA_OPENCTM_SRC_DIR}
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
