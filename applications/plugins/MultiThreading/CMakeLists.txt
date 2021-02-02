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
    )

set(SOURCE_FILES
    src/MultiThreading/initMultiThreading.cpp
    src/MultiThreading/AnimationLoopParallelScheduler.cpp
    src/MultiThreading/AnimationLoopTasks.cpp
    src/MultiThreading/BeamLinearMapping_mt.cpp
    src/MultiThreading/DataExchange.cpp
    src/MultiThreading/MeanComputation.cpp
    )

find_package(SofaMiscMapping REQUIRED)
find_package(SofaSimulationCommon REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping SofaConstraint SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")


## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
    )
