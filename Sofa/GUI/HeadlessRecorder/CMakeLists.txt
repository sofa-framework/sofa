cmake_minimum_required(VERSION 3.22)
project(Sofa.GUI.HeadlessRecorder)

find_package(Sofa.GL REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)

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

set(SRC_ROOT src/sofa/gui/headlessrecorder)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/HeadlessRecorder.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/HeadlessRecorder.cpp
)

if(SOFA_BUILD_TESTS)
    configure_file(${SRC_ROOT}/headlessRecorder_test.sh ${CMAKE_BINARY_DIR}/bin/headlessRecorder_test COPYONLY)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Common Sofa.GL)
target_link_libraries(${PROJECT_NAME} PUBLIC ${X11_LIBRARIES})

# Create build and install versions of .ini file for resources finding
set(FFMPEG_EXEC_PATH "${FFMPEG_EXEC_FILE}") # FFMPEG_EXEC_FILE is set by FindFFMEG_exec.cmake
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
get_filename_component(FFMPEG_EXEC_FILENAME "${FFMPEG_EXEC_FILE}" NAME)
set(FFMPEG_EXEC_PATH "../bin/${FFMPEG_EXEC_FILENAME}") # relative path for install dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
)
