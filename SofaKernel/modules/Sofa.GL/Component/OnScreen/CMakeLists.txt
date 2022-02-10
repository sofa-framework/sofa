cmake_minimum_required(VERSION 3.12)
project(Sofa.GL.Component.OnScreen LANGUAGES CXX)

set(SOFAGLCOMPONENTONSCREEN_SOURCE_DIR "src/sofa/gl/component/onscreen")

set(HEADER_FILES
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/config.h.in
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/init.h
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglColorMap.h
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglLabel.h
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglViewport.h
)

set(SOURCE_FILES
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/init.cpp
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglColorMap.cpp
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglLabel.cpp
    ${SOFAGLCOMPONENTONSCREEN_SOURCE_DIR}/OglViewport.cpp
)

sofa_find_package(SofaSimulationCore REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(SofaBaseVisual REQUIRED)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore Sofa.GL Sofa.Component.Visual SofaBaseVisual)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
