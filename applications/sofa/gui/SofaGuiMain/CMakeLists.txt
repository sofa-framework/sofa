cmake_minimum_required(VERSION 3.1)
project(SofaGuiMain)

set(HEADER_FILES
    ../Main.h
)

set(SOURCE_FILES
    ../Main.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)
if(SOFAGUI_HAVE_SOFAGUIQT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiQt)
endif()
if(SOFAGUI_HAVE_SOFAHEADLESSRECORDER)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaHeadlessRecorder)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GUIMAIN")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGui SofaGuiMain "SofaGui/sofa/gui")
