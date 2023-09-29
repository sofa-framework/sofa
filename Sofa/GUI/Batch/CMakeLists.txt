cmake_minimum_required(VERSION 3.12)
project(Sofa.GUI.Batch LANGUAGES CXX)

set(SOFAGUIBATCH_ROOT src/sofa/gui/batch)

set(HEADER_FILES
    ${SOFAGUIBATCH_ROOT}/config.h.in
    ${SOFAGUIBATCH_ROOT}/init.h
    ${SOFAGUIBATCH_ROOT}/BatchGUI.h
    ${SOFAGUIBATCH_ROOT}/ProgressBar.h
    ${SOFAGUIBATCH_ROOT}/indicators/indicators.hpp
)

set(SOURCE_FILES
    ${SOFAGUIBATCH_ROOT}/init.cpp
    ${SOFAGUIBATCH_ROOT}/BatchGUI.cpp
    ${SOFAGUIBATCH_ROOT}/ProgressBar.cpp
)

sofa_find_package(Sofa.GUI.Common REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Common)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
)
