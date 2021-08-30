cmake_minimum_required(VERSION 3.12)
project(Sofa.LinearAlgebra LANGUAGES CXX)

set(SOFALINEARALGEBRASRC_ROOT "src/sofa/linearalgebra")

set(HEADER_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/config.h.in
    ${SOFALINEARALGEBRASRC_ROOT}/init.h

)

set(SOURCE_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/init.cpp
)

sofa_find_package(Sofa.Type REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Helper)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
