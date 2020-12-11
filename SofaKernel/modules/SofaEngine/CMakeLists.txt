cmake_minimum_required(VERSION 3.12)
project(SofaEngine LANGUAGES CXX)

set(SOFAENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAENGINE_SRC}/config.h.in
    ${SOFAENGINE_SRC}/initSofaEngine.h
    ${SOFAENGINE_SRC}/BoxROI.h
    ${SOFAENGINE_SRC}/BoxROI.inl
)

set(SOURCE_FILES
    ${SOFAENGINE_SRC}/initSofaEngine.cpp
    ${SOFAENGINE_SRC}/BoxROI.cpp
)

find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaCore SofaDefaultType)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAENGINE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
