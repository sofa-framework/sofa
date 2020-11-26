cmake_minimum_required(VERSION 3.12)
project(Sofa.Types LANGUAGES CXX)

set(SRC_ROOT "src/sofa/types")

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_SOFATYPES) # To remove once sofa_add_targets_to_package remove the dot in the generated definition

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )

## Tests
# cmake_dependent_option(SOFATYPES_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFATYPES_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()


