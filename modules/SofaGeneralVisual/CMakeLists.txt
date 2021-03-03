cmake_minimum_required(VERSION 3.12)
project(SofaGeneralVisual LANGUAGES CXX)

find_package(SofaBase REQUIRED)

set(SOFAGENERALVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALVISUAL_SRC}/config.h.in
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.h
    ${SOFAGENERALVISUAL_SRC}/RecordedCamera.h
    ${SOFAGENERALVISUAL_SRC}/VisualTransform.h
    ${SOFAGENERALVISUAL_SRC}/Visual3DText.h
    )

set(SOURCE_FILES
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.cpp
    ${SOFAGENERALVISUAL_SRC}/RecordedCamera.cpp
    ${SOFAGENERALVISUAL_SRC}/VisualTransform.cpp
    ${SOFAGENERALVISUAL_SRC}/Visual3DText.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseVisual)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
