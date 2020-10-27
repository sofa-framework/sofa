cmake_minimum_required(VERSION 3.12)
project(SofaTypes VERSION 1.0 LANGUAGES CXX)

set(SRC_ROOT "src/sofa/types")

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/Vec.h
    ${SRC_ROOT}/std/fixed_array.h
    ${SRC_ROOT}/std/fixed_array_algorithms.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/Vec.cpp
    ${SRC_ROOT}/std/fixed_array.cpp
)

set(COMPAT_HEADER_FILES
    src/sofa/defaulttype/Vec.h
    src/sofa/helper/fixed_array.h
    src/sofa/helper/fixed_array_algorithms.h
)

source_group("compat" ${COMPAT_HEADER_FILES})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${COMPAT_HEADER_FILES} ${SOURCE_FILES})

# TO REMOVE AS SOON AS POSSIBLE
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../SofaDefaultType/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../SofaHelper/src>")

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )


# if(SOFA_BUILD_TESTS)
#     add_subdirectory(SofaTypes_test)
# endif()

