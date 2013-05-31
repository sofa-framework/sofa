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

AddCompilerDefinitions("SOFA_BUILD_ARTRACKPLUGIN")
AddLinkerDependencies(ARTrackLib SofaUserInteraction)

if(WIN32)
	AddLinkerDependencies(Ws2_32)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)