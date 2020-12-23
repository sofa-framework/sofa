cmake_minimum_required(VERSION 3.12)
project(SofaBaseVisual LANGUAGES CXX)

set(SOFABASEVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEVISUAL_SRC}/config.h.in
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.h
    ${SOFABASEVISUAL_SRC}/BaseCamera.h
    ${SOFABASEVISUAL_SRC}/BackgroundSetting.h
    ${SOFABASEVISUAL_SRC}/Camera.h
    ${SOFABASEVISUAL_SRC}/InteractiveCamera.h
    ${SOFABASEVISUAL_SRC}/VisualModelImpl.h
    ${SOFABASEVISUAL_SRC}/VisualStyle.h
)

set(SOURCE_FILES
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.cpp
    ${SOFABASEVISUAL_SRC}/BaseCamera.cpp
    ${SOFABASEVISUAL_SRC}/BackgroundSetting.cpp
    ${SOFABASEVISUAL_SRC}/Camera.cpp
    ${SOFABASEVISUAL_SRC}/InteractiveCamera.cpp
    ${SOFABASEVISUAL_SRC}/VisualModelImpl.cpp
    ${SOFABASEVISUAL_SRC}/VisualStyle.cpp
)

sofa_find_package(TinyXML REQUIRED)
find_package(SofaBaseTopology REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml) # Private because not exported in API

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASETOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASEVISUAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
