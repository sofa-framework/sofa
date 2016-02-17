cmake_minimum_required(VERSION 2.8.12)
project(SofaGeneralVisual)

set(HEADER_FILES
    RecordedCamera.h
    VisualTransform.h
    config.h
    initGeneralVisual.h
)

set(SOURCE_FILES
    RecordedCamera.cpp
    VisualTransform.cpp
    initGeneralVisual.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseVisual)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
