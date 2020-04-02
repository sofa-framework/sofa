cmake_minimum_required(VERSION 3.1)
project(SofaBaseUtils)

find_package(SofaFramework REQUIRED)

set(SOFABASEUTILS_SRC .)

set(HEADER_FILES
    config.h
    initBaseUtils.h
    ${SOFABASEUTILS_SRC}/messageHandlerComponent.h
    ${SOFABASEUTILS_SRC}/MakeAliasComponent.h
    ${SOFABASEUTILS_SRC}/MakeDataAliasComponent.h
    ${SOFABASEUTILS_SRC}/InfoComponent.h
    ${SOFABASEUTILS_SRC}/RequiredPlugin.h
    ${SOFABASEUTILS_SRC}/AddResourceRepository.h
)

set(SOURCE_FILES
    initBaseUtils.cpp
    ${SOFABASEUTILS_SRC}/messageHandlerComponent.cpp
    ${SOFABASEUTILS_SRC}/MakeAliasComponent.cpp
    ${SOFABASEUTILS_SRC}/MakeDataAliasComponent.cpp
    ${SOFABASEUTILS_SRC}/InfoComponent.cpp
    ${SOFABASEUTILS_SRC}/RequiredPlugin.cpp
    ${SOFABASEUTILS_SRC}/AddResourceRepository.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_UTILS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} "SofaBase/${PROJECT_NAME}")

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFABASEUTILS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFABASEUTILS_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(SofaBase_test)
# endif()
