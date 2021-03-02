cmake_minimum_required(VERSION 3.12)
project(SofaMiscEngine LANGUAGES CXX)

find_package(SofaNonUniformFem REQUIRED)

set(SOFAMISCENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCENGINE_SRC}/config.h.in
    ${SOFAMISCENGINE_SRC}/initSofaMiscEngine.h
    )

set(SOURCE_FILES
    ${SOFAMISCENGINE_SRC}/initSofaMiscEngine.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.h
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.inl
    ${SOFAMISCENGINE_SRC}/Distances.h
    ${SOFAMISCENGINE_SRC}/Distances.inl
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.h
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.cpp
    ${SOFAMISCENGINE_SRC}/Distances.cpp
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaNonUniformFem)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )

