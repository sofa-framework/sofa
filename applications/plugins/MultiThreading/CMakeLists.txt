cmake_minimum_required(VERSION 2.8.12)
project(MultiThreading)


set(HEADER_FILES
    src/DataExchange.h
    src/DataExchange.inl
    config.h
    # src/Observer.h
)

set(SOURCE_FILES
    src/DataExchange.cpp
    # src/Observer.cpp
    src/initMultiThreading.cpp
)

find_package(SofaMisc REQUIRED)
find_package(Boost QUIET COMPONENTS thread)

if(Boost_FOUND)
    list(APPEND HEADER_FILES
        src/AnimationLoopParallelScheduler.h
        src/AnimationLoopTasks.h
        src/BeamLinearMapping_mt.h
        src/BeamLinearMapping_mt.inl
        src/BeamLinearMapping_tasks.inl
        # src/ParallelForTask.h
        src/TaskSchedulerBoost.h
        src/Tasks.h
        src/Tasks.inl)

    list(APPEND SOURCE_FILES
        src/AnimationLoopParallelScheduler.cpp
        src/AnimationLoopTasks.cpp
        src/BeamLinearMapping_mt.cpp
        # ParallelForTask.cpp
        src/TaskSchedulerBoost.cpp
        src/Tasks.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

## uncomment this to use the intel threading building block library
#set(SOFA_HAVE_TBB true)

# # Useful ?
# find_package(SofaCUDA)
# if(SofaCUDA_FOUND)
#     target_link_libraries(${PROJECT_NAME} SofaCUDA)
# endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT MultiThreading_libraries
        EXPORT MultiThreadingTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
