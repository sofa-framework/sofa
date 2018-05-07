cmake_minimum_required(VERSION 3.1)
project(SofaHeadlessRecorder)
message("-- HeadlessRecorder is a 17.12 experimental feature, use it at your own risk.")

if(UNIX AND NOT APPLE)
    message("HeadlessRecorder for linux")
else()
    message(FATAL_ERROR "${PROJECT_NAME} is a Linux only feature.")
endif()


# FFMPEG
find_package(FFMPEG QUIET)
if(FFMPEG_FOUND)
    message("Found FFMPEG/LibAV libraries")
    include_directories(${FFMPEG_INCLUDE_DIR})
else()
    message(SEND_ERROR "Can't find libavcodec, libavformat or libavutil.")
endif()
set(SOFA_HAVE_AVCODEC ${AVCODEC_FOUND})

# X11
find_package(X11 QUIET)
if(X11_FOUND)
    message("Found X11 libraries")
    include_directories(${X11_INCLUDE_DIR})
else()
    message(SEND_ERROR "Can't find X11 libraries.")
endif()


set(HEADER_FILES
    HeadlessRecorder.h)

set(SOURCE_FILES
    HeadlessRecorder.cpp)

if(SOFAGUI_BUILD_TESTS)
    configure_file(headlessRecorder_test ${CMAKE_BINARY_DIR}/bin/headlessRecorder_test COPYONLY)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)
target_link_libraries(${PROJECT_NAME} PUBLIC ${FFMPEG_LIBRARIES} ${X11_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIHEADLESSRECORDER")

sofa_install_targets(SofaGui SofaHeadlessRecorder "")
