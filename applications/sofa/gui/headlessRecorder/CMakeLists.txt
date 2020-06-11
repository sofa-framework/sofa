cmake_minimum_required(VERSION 3.1)
project(SofaHeadlessRecorder)

# FFMPEG
find_package(FFMPEG_exec)

# X11
sofa_find_package(X11 QUIET)
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


if(SOFA_BUILD_TESTS)
    configure_file(headlessRecorder_test ${CMAKE_BINARY_DIR}/bin/headlessRecorder_test COPYONLY)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)
target_link_libraries(${PROJECT_NAME} PUBLIC ${X11_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIHEADLESSRECORDER")

# Create build and install versions of .ini file for resources finding
set(FFMPEG_EXEC_PATH "${FFMPEG_EXEC_FILE}") # FFMPEG_EXEC_FILE is set by FindFFMEG_exec.cmake
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
get_filename_component(FFMPEG_EXEC_FILENAME "${FFMPEG_EXEC_FILE}" NAME)
set(FFMPEG_EXEC_PATH "../bin/${FFMPEG_EXEC_FILENAME}") # relative path for install dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

sofa_install_targets(SofaGui SofaHeadlessRecorder "")
