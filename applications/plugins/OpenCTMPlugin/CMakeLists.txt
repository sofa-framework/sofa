cmake_minimum_required(VERSION 2.8.12)
projetc(OpenCTMPlugin)

set(HEADER_FILES
    OpenCTMExporter.h
    OpenCTMLoader.h
    config.h
)

set(SOURCE_FILES
    OpenCTMExporter.cpp
    OpenCTMLoader.cpp
    initOpenCTMPlugin.cpp
)

set(README_FILES OpenCTMPlugin.txt)

find_package(OpenCTM REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
include_directories("${OPENCTM_INCLUDE_PATH}")
add_definitions("-DSOFA_BUILD_OPENCTMPLUGIN")
target_linker_libraries(${PROJECT_NAME} SofaBaseTopology ${OPENCTM_LIBRARY})
