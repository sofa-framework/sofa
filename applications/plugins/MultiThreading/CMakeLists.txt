cmake_minimum_required(VERSION 2.8)
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

    list(APPEND HEADER_FILES "src/TaskSchedulerBoost.h")
    list(APPEND SOURCE_FILES "src/TaskSchedulerBoost.cpp")

    list(APPEND HEADER_FILES "src/Tasks.h")
    list(APPEND HEADER_FILES "src/Tasks.inl")
    list(APPEND SOURCE_FILES "src/Tasks.cpp")


    list(APPEND HEADER_FILES "src/AnimationLoopParallelScheduler.h")
    list(APPEND SOURCE_FILES "src/AnimationLoopParallelScheduler.cpp")

    list(APPEND HEADER_FILES "src/AnimationLoopTasks.h")
    list(APPEND SOURCE_FILES "src/AnimationLoopTasks.cpp")

    #list(APPEND HEADER_FILES "src/ParallelForTask.h")
    #list(APPEND SOURCE_FILES "src/ParallelForTask.cpp")

    list(APPEND HEADER_FILES "src/BeamLinearMapping_mt.h")
    list(APPEND HEADER_FILES "src/BeamLinearMapping_mt.inl")
    list(APPEND SOURCE_FILES "src/BeamLinearMapping_mt.cpp")
    list(APPEND HEADER_FILES "src/BeamLinearMapping_tasks.inl")

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
# target_link_libraries(${PROJECT_NAME} SofaCUDA)
# endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT MultiThreading_libraries
        EXPORT MultiThreadingTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
