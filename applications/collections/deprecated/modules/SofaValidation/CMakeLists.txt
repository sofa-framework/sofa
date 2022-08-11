cmake_minimum_required(VERSION 3.12)
project(SofaValidation LANGUAGES CXX)

set(SOFAVALIDATION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAVALIDATION_SRC}/config.h.in
    ${SOFAVALIDATION_SRC}/initSofaValidation.h
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
    ${SOFAVALIDATION_SRC}/initSofaValidation.cpp
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

find_package(Sofa.Simulation.Common REQUIRED)
sofa_find_package(Sofa.Component.Collision REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Common Sofa.Component.Collision)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. For CompareState and CompareTopology, use Sofa.Component.Playback instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Playback REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Playback)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)

## Tests
cmake_dependent_option(SOFAVALIDATION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAVALIDATION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaValidation_test)
endif()

