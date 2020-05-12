cmake_minimum_required(VERSION 3.1)
project(SofaValidation VERSION 1.0 LANGUAGES CXX)

find_package(SofaBase REQUIRED)
find_package(SofaCommon REQUIRED)
find_package(SofaGeneral REQUIRED)

set(SOFAVALIDATION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAVALIDATION_SRC}/config.h.in
    ${SOFAVALIDATION_SRC}/initValidation.h
    ${SOFAVALIDATION_SRC}/CompareState.h
    ${SOFAVALIDATION_SRC}/CompareTopology.h
    ${SOFAVALIDATION_SRC}/DataController.h
    ${SOFAVALIDATION_SRC}/DataMonitor.h
    ${SOFAVALIDATION_SRC}/DevAngleCollisionMonitor.h
    ${SOFAVALIDATION_SRC}/DevAngleCollisionMonitor.inl
    ${SOFAVALIDATION_SRC}/DevMonitor.h
    ${SOFAVALIDATION_SRC}/DevMonitorManager.h
    ${SOFAVALIDATION_SRC}/DevTensionMonitor.h
    ${SOFAVALIDATION_SRC}/DevTensionMonitor.inl
    ${SOFAVALIDATION_SRC}/EvalPointsDistance.h
    ${SOFAVALIDATION_SRC}/EvalPointsDistance.inl
    ${SOFAVALIDATION_SRC}/EvalSurfaceDistance.h
    ${SOFAVALIDATION_SRC}/EvalSurfaceDistance.inl
    ${SOFAVALIDATION_SRC}/ExtraMonitor.h
    ${SOFAVALIDATION_SRC}/ExtraMonitor.inl
    ${SOFAVALIDATION_SRC}/Monitor.h
    ${SOFAVALIDATION_SRC}/Monitor.inl
    )

set(SOURCE_FILES
    ${SOFAVALIDATION_SRC}/initValidation.cpp
    ${SOFAVALIDATION_SRC}/CompareState.cpp
    ${SOFAVALIDATION_SRC}/CompareTopology.cpp
    ${SOFAVALIDATION_SRC}/DataController.cpp
    ${SOFAVALIDATION_SRC}/DataMonitor.cpp
    ${SOFAVALIDATION_SRC}/DevAngleCollisionMonitor.cpp
    ${SOFAVALIDATION_SRC}/DevMonitorManager.cpp
    ${SOFAVALIDATION_SRC}/DevTensionMonitor.cpp
    ${SOFAVALIDATION_SRC}/EvalPointsDistance.cpp
    ${SOFAVALIDATION_SRC}/EvalSurfaceDistance.cpp
    ${SOFAVALIDATION_SRC}/ExtraMonitor.cpp
    ${SOFAVALIDATION_SRC}/Monitor.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseCollision SofaLoader SofaMeshCollision SofaGeneralLoader)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAVALIDATION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "${SOFAVALIDATION_SRC}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

## Tests
cmake_dependent_option(SOFAVALIDATION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAVALIDATION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaValidation_test)
endif()

