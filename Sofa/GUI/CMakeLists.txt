cmake_minimum_required(VERSION 3.12)
project(Sofa.GUI LANGUAGES CXX)

set(SOFAGUI_SOURCE_DIR "src/sofa/gui")

set(SOFAGUI_DIRS Component Common Batch Qt)
set(SOFAGUI_TARGETS)
foreach(dir ${SOFAGUI_DIRS})
    sofa_add_module(${dir} ${PROJECT_NAME}.${dir} ON)
    if(TARGET ${PROJECT_NAME}.${dir})
        list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.${dir})
    endif()
endforeach()

sofa_add_module(HeadlessRecorder ${PROJECT_NAME}.HeadlessRecorder OFF)
if(TARGET ${PROJECT_NAME}.HeadlessRecorder)
    list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.HeadlessRecorder)
endif()

set(HEADER_FILES
    ${SOFAGUI_SOURCE_DIR}/config.h.in
    ${SOFAGUI_SOURCE_DIR}/init.h
)
set(SOURCE_FILES
    ${SOFAGUI_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGUI_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
