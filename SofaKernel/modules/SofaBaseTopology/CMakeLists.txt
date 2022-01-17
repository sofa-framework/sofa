cmake_minimum_required(VERSION 3.12)
project(SofaBaseTopology LANGUAGES CXX)

set(SOFABASETOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/config.h.in
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.h
)

set(SOURCE_FILES
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.cpp
)

sofa_find_package(SofaFramework REQUIRED)
sofa_find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

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
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASETOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASETOPOLOGY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
