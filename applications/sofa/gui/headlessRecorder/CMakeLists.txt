cmake_minimum_required(VERSION 3.1)
project(SofaHeadlessRecorder)

set(HEADER_FILES
    HeadlessRecorder.h)

set(SOURCE_FILES
    HeadlessRecorder.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} -lpng -lEGL -lavcodec -lswscale -lavutil SofaGuiCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIHEADLESSRECORDER")

sofa_install_targets(SofaGui SofaHeadlessRecorder "")
