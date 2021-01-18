cmake_minimum_required(VERSION 3.12)
project(SofaGuiMain)

find_package(SofaGuiCommon)
sofa_find_package(SofaGuiQt QUIET)
sofa_find_package(SofaHeadlessRecorder QUIET)

set(HEADER_FILES
    src/sofa/gui/Main.h
)

set(SOURCE_FILES
    src/sofa/gui/Main.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon)

if(SofaGuiQt_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiQt)
endif()
if(SofaHeadlessRecorder_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaHeadlessRecorder)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaGui")
