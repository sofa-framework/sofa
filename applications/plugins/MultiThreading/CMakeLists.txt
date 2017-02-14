cmake_minimum_required(VERSION 3.1)
project(MultiThreading)

set(MULTITHREADING_MAJOR_VERSION 0)
set(MULTITHREADING_MINOR_VERSION 1)
set(MULTITHREADING_VERSION ${MULTITHREADING_MAJOR_VERSION}.${MULTITHREADING_MINOR_VERSION})

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
find_package(Boost QUIET COMPONENTS system thread date_time chrono)

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
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${MULTITHREADING_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

## uncomment this to use the intel threading building block library
#set(SOFA_HAVE_TBB true)

# # Useful ?
# find_package(SofaCUDA)
# if(SofaCUDA_FOUND)
#     target_link_libraries(${PROJECT_NAME} SofaCUDA)
# endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(MultiThreading ${MULTITHREADING_VERSION} MultiThreading MultiThreading)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

# TODO replace boost::thread by std::thread in this plugin
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()
