cmake_minimum_required(VERSION 3.12)
project(MultiThreading VERSION 0.1)

set(HEADER_FILES
    config.h
    src/AnimationLoopParallelScheduler.h
    src/AnimationLoopTasks.h
    src/BeamLinearMapping_mt.h
    src/BeamLinearMapping_mt.inl
    src/BeamLinearMapping_tasks.inl
    src/DataExchange.h
    src/DataExchange.inl
    src/MeanComputation.h
    src/MeanComputation.inl
    )

set(SOURCE_FILES
    src/initMultiThreading.cpp
    src/AnimationLoopParallelScheduler.cpp
    src/AnimationLoopTasks.cpp
    src/BeamLinearMapping_mt.cpp
    src/DataExchange.cpp
    src/MeanComputation.cpp
    )

find_package(SofaMisc REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping SofaConstraint)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")


## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
