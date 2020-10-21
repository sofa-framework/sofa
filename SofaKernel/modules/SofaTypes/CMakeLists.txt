cmake_minimum_required(VERSION 3.12)
project(SofaTypes VERSION 1.0 LANGUAGES CXX)

# find_package(SofaHelper REQUIRED)

set(SRC_ROOT "src/sofa/types")

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )


# if(SOFA_BUILD_TESTS)
#     add_subdirectory(SofaTypes_test)
# endif()

