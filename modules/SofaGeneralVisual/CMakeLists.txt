cmake_minimum_required(VERSION 3.12)
project(SofaGeneralVisual)

set(HEADER_FILES
    config.h
    initGeneralVisual.h
    )

set(SOURCE_FILES
    initGeneralVisual.cpp
    )

list(APPEND HEADER_FILES
    RecordedCamera.h
    VisualTransform.h
    Visual3DText.h
    )
list(APPEND SOURCE_FILES
    RecordedCamera.cpp
    VisualTransform.cpp
    Visual3DText.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseVisual)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
