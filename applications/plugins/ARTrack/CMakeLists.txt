cmake_minimum_required(VERSION 3.1)
project(ARTrack)

add_subdirectory(extlibs/ARTrackLib ARTrackLib)
set_target_properties(ARTrackLib PROPERTIES DEBUG_POSTFIX "_d")

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

find_package(ARTrackLib REQUIRED)
find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaUserInteraction SofaGeneralRigid)
target_link_libraries(${PROJECT_NAME} ARTrackLib)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} Ws2_32)
endif()

if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT ARTrack_libraries
        EXPORT ARTrackTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
