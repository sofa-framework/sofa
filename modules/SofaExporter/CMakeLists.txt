cmake_minimum_required(VERSION 3.12)
project(SofaExporter LANGUAGES CXX)

sofa_find_package(SofaSimulationCore REQUIRED)
# sofa_find_package(ZLIB REQUIRED)

set(SRC_ROOT src/SofaExporter)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaExporter.h
    )

set(SOURCE_FILES
    ${SRC_ROOT}/initSofaExporter.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)
# target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
# if(CMAKE_SYSTEM_NAME STREQUAL Windows)
#     sofa_install_libraries(TARGETS ZLIB::ZLIB)
# endif()

sofa_create_package_with_targets(
    PACKAGE_NAME SofaExporter
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaExporter"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAEXPORTER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAEXPORTER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaExporter_test)
endif()
