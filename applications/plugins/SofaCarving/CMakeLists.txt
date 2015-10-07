cmake_minimum_required(VERSION 2.8.12)
project(SofaCarving)


set(HEADER_FILES
    CarvingManager.h
    config.h
)

set(SOURCE_FILES
    CarvingManager.cpp
    initSofaCarving.cpp
)

set(README_FILES SofaCarving.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACARVING")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
