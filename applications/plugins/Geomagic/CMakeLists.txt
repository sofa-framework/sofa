cmake_minimum_required(VERSION 2.8.12)
project(Geomagic)

find_package(SofaHaptics REQUIRED)
find_package(OpenHaptics)
if( NOT OPENHAPTICS_FOUND )
    message( SEND_ERROR "OpenHaptics library not found, GeomagicDriver cannot compile")
endif()

find_package(SofaOpenglVisual REQUIRED)

set(HEADER_FILES
      src/GeomagicDriver.h
      src/initPlugin.h
      src/config.h
)

set(SOURCE_FILES
      src/GeomagicDriver.cpp
      src/initPlugin.cpp
)

set(SCENES_FILES
      scenes/Geomagic-Demo.scn
)

include_directories( ${OPENHAPTICS_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${SCENES_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DPLUGIN_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GEOMAGIC")
target_link_libraries(${PROJECT_NAME} SofaHelper SofaUserInteraction SofaHaptics SofaOpenglVisual ${OPENHAPTICS_LIBRARIES})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

## Install rules for the library; CMake package configurations files
sofa_create_package(${PROJECT_NAME} 1.0 ${PROJECT_NAME} ${PROJECT_NAME})
