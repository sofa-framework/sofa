cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
set(LINKER_DEPENDENCIES ARTrackLib SofaUserInteraction)

if(WIN32)
	set(LINKER_DEPENDENCIES ${LINKER_DEPENDENCIES} Ws2_32)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)