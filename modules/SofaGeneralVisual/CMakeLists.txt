cmake_minimum_required(VERSION 3.12)
project(SofaGeneralVisual LANGUAGES CXX)

find_package(SofaBase REQUIRED)

set(SOFAGENERALVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALVISUAL_SRC}/config.h.in
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.h
    )

set(SOURCE_FILES
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALVISUAL_SRC}/RecordedCamera.h
    ${SOFAGENERALVISUAL_SRC}/VisualTransform.h
    ${SOFAGENERALVISUAL_SRC}/Visual3DText.h
    )
list(APPEND SOURCE_FILES
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
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFAGENERALVISUAL_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFAGENERALVISUAL_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()
