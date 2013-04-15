cmake_minimum_required(VERSION 2.8)

project("ARTrack")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package(ARTrackLibs REQUIRED)
include_directories(${ARTRACKLIBS_INCLUDE_DIRS})

set(HEADER_FILES

    ARTrackDriver.h 
    ARTrackEvent.h 
    ARTrackController.h 
    ARTrackController.inl

	)

set(SOURCE_FILES

    ARTrackDriver.cpp 
    initARTrack.cpp 
    ARTrackEvent.cpp 
    ARTrackController.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_ARTRACKPLUGIN")
#set(LINKER_DEPENDENCIES )

include(${SOFA_CMAKE_DIR}/postProject.cmake)