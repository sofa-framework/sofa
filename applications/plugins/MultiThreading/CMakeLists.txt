cmake_minimum_required(VERSION 3.12)
project(MultiThreading VERSION 0.1)

set(HEADER_FILES
    src/MultiThreading/config.h
    src/MultiThreading/AnimationLoopParallelScheduler.h
    src/MultiThreading/AnimationLoopTasks.h
    src/MultiThreading/BeamLinearMapping_mt.h
    src/MultiThreading/BeamLinearMapping_mt.inl
    src/MultiThreading/BeamLinearMapping_tasks.inl
    src/MultiThreading/DataExchange.h
    src/MultiThreading/DataExchange.inl
    src/MultiThreading/MeanComputation.h
    src/MultiThreading/MeanComputation.inl
    src/MultiThreading/ParallelBruteForceBroadPhase.h
    src/MultiThreading/ParallelBVHNarrowPhase.h
    src/MultiThreading/ParallelHexahedronFEMForceField.h
    src/MultiThreading/ParallelHexahedronFEMForceField.inl
    )

set(SOURCE_FILES
    src/MultiThreading/initMultiThreading.cpp
    src/MultiThreading/AnimationLoopParallelScheduler.cpp
    src/MultiThreading/AnimationLoopTasks.cpp
    src/MultiThreading/BeamLinearMapping_mt.cpp
    src/MultiThreading/DataExchange.cpp
    src/MultiThreading/MeanComputation.cpp
    src/MultiThreading/ParallelBruteForceBroadPhase.cpp
    src/MultiThreading/ParallelBVHNarrowPhase.cpp
    src/MultiThreading/ParallelHexahedronFEMForceField.cpp
    )

find_package(Sofa.Simulation.Common REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Common)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.FEM.Elastic)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Mapping.Linear)
target_link_libraries(${PROJECT_NAME} Sofa.Component.StateContainer)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")


## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(MULTITHREADING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(MULTITHREADING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
