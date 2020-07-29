cmake_minimum_required(VERSION 3.12)
project(SofaGuiMain)

set(HEADER_FILES
    src/sofa/gui/Main.h
)

set(SOURCE_FILES
    src/sofa/gui/Main.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)
if(SOFAGUI_HAVE_SOFAGUIQT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiQt)
endif()
if(SOFAGUI_HAVE_SOFAHEADLESSRECORDER)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaHeadlessRecorder)
endif()

sofa_install_targets(SofaGui SofaGuiMain "SofaGui" "src")
