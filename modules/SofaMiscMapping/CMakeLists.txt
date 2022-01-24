cmake_minimum_required(VERSION 3.12)
project(SofaMiscMapping LANGUAGES CXX)

set(SOFAMISCMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCMAPPING_SRC}/config.h.in
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.h
    )

set(SOURCE_FILES
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.cpp
    )

sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaRigid REQUIRED) 
sofa_find_package(SofaGeneralSimpleFem REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaRigid SofaGeneralSimpleFem)

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

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCMAPPING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCMAPPING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscMapping_test)
endif()