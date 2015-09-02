include(${SOFA_CMAKE_DIR}/preProject.cmake)

cmake_minimum_required(VERSION 2.8)

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

if(SOFA-EXTERNAL_BOOST)
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

## uncomment this to use the intel threading building block library
#set(SOFA_HAVE_TBB true)



set(GROUP_BASE_DIR "src")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})


AddCompilerDefinitions("SOFA_MULTITHREADING_PLUGIN" )
AddLinkerDependencies(SofaComponentMain SofaMiscMapping)

if(SOFA-PLUGIN_SOFACUDA)
    AddLinkerDependencies(SofaCUDA)
endif()



#
# Organize projects into sofa-asclepios folders
#
#SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
#SET_PROPERTY(TARGET ${PROJECT_NAME} PROPERTY FOLDER "sofa-asclepios")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
