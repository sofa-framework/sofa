include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    ARTrackController.h
    ARTrackController.inl
    ARTrackDriver.h
    ARTrackEvent.h
)

set(SOURCE_FILES
    ARTrackController.cpp
    ARTrackDriver.cpp
    ARTrackEvent.cpp
    initARTrack.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_ARTRACKPLUGIN")
AddLinkerDependencies(ARTrackLib SofaUserInteraction)

if(WIN32)
    AddLinkerDependencies(Ws2_32)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)