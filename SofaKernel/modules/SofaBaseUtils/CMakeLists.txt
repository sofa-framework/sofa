cmake_minimum_required(VERSION 3.12)
project(SofaBaseUtils LANGUAGES CXX)

set(SOFABASEUTILS_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEUTILS_SRC}/config.h.in
    ${SOFABASEUTILS_SRC}/initSofaBaseUtils.h
    ${SOFABASEUTILS_SRC}/messageHandlerComponent.h
    ${SOFABASEUTILS_SRC}/MakeAliasComponent.h
    ${SOFABASEUTILS_SRC}/MakeDataAliasComponent.h
    ${SOFABASEUTILS_SRC}/InfoComponent.h
    ${SOFABASEUTILS_SRC}/RequiredPlugin.h
    ${SOFABASEUTILS_SRC}/AddResourceRepository.h
)

set(SOURCE_FILES
    ${SOFABASEUTILS_SRC}/initSofaBaseUtils.cpp
    ${SOFABASEUTILS_SRC}/messageHandlerComponent.cpp
    ${SOFABASEUTILS_SRC}/MakeAliasComponent.cpp
    ${SOFABASEUTILS_SRC}/MakeDataAliasComponent.cpp
    ${SOFABASEUTILS_SRC}/InfoComponent.cpp
    ${SOFABASEUTILS_SRC}/RequiredPlugin.cpp
    ${SOFABASEUTILS_SRC}/AddResourceRepository.cpp
)

find_package(SofaFramework REQUIRED) # SofaCore

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASEUTILS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASEUTILS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaBaseUtils_test)
endif()
