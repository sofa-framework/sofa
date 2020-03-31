cmake_minimum_required(VERSION 3.1)
project(SofaMiscEngine VERSION 1.0 LANGUAGES CXX)

find_package(SofaAdvanced REQUIRED)

set(SOFAMISCENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCENGINE_SRC}/config.h.in
    ${SOFAMISCENGINE_SRC}/initMiscEngine.h
    )

set(SOURCE_FILES
    ${SOFAMISCENGINE_SRC}/initMiscEngine.cpp
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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_ENGINE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/${PROJECT_NAME}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
