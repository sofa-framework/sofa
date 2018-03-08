cmake_minimum_required(VERSION 3.1)
project(SofaHeadlessRecorder)
message("-- HeadlessRecorder is a 17.12 experimental feature, use it at your own risk.")


set(HEADER_FILES
    HeadlessRecorder.h)

set(SOURCE_FILES
    HeadlessRecorder.cpp)


if(UNIX AND NOT APPLE)
    message("HeadlessRecorder for linux")
    if(SOFAGUI_BUILD_TESTS)
        configure_file(headlessRecorder_test ${CMAKE_BINARY_DIR}/bin/headlessRecorder_test COPYONLY)
    endif()

    ###################################################################### FFMPEG
    find_package(FFMPEG REQUIRED)
    if (FFMPEG_FOUND)
        message("Found FFMPEG/LibAV libraries")
        include_directories(${FFMPEG_INCLUDE_DIR})
    else (FFMPEG_FOUND)
        message("Can't find libavcodec, libavformat or libavutil. Add them!")
    endif (FFMPEG_FOUND)
    set(SOFA_HAVE_AVCODEC ${AVCODEC_FOUND})

    ###################################################################### X11
    find_package(X11 REQUIRED)
    if (X11_FOUND)
        message("Found X11 libraries")
        include_directories(${X11_INCLUDE_DIR})
    else (X11_FOUND)
        message("Can't find X11 libraries")
    endif (X11_FOUND)

    add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

    target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${FFMPEG_LIBRARIES} ${X11_LIBRARIES})
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIHEADLESSRECORDER")

    sofa_install_targets(SofaGui SofaHeadlessRecorder "")
endif()
