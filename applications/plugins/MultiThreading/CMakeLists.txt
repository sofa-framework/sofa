cmake_minimum_required(VERSION 3.1)
project(MultiThreading)

set(MULTITHREADING_MAJOR_VERSION 0)
set(MULTITHREADING_MINOR_VERSION 1)
set(MULTITHREADING_VERSION ${MULTITHREADING_MAJOR_VERSION}.${MULTITHREADING_MINOR_VERSION})

set(HEADER_FILES
    src/TaskScheduler.h
    src/Tasks.h
    src/Locks.h
    src/AnimationLoopParallelScheduler.h
    src/AnimationLoopTasks.h
    src/BeamLinearMapping_mt.h
    src/BeamLinearMapping_mt.inl
    src/BeamLinearMapping_tasks.inl
    src/DataExchange.h
    src/DataExchange.inl
    
)

set(SOURCE_FILES
    src/initMultiThreading.cpp
    src/TaskScheduler.cpp
    src/Tasks.cpp
    src/AnimationLoopParallelScheduler.cpp
    src/AnimationLoopTasks.cpp
    src/BeamLinearMapping_mt.cpp
    src/DataExchange.cpp    
)

find_package(SofaMisc REQUIRED)
find_package(Boost QUIET COMPONENTS system thread date_time chrono)

find_package(image QUIET)
IF(image_FOUND)
set (WITH_IMAGE TRUE)
ELSE(image_FOUND)
set (WITH_IMAGE FALSE)
ENDIF()

## Multithreading/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/MultiThreading/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/MultiThreading/config.h" DESTINATION "include/MultiThreading")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
IF(image_FOUND)
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping image)
ELSE(image_FOUND)
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaMiscMapping)
ENDIF()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${MULTITHREADING_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")


## Install rules for the library and headers; CMake package configurations files
sofa_create_package(MultiThreading ${MULTITHREADING_VERSION} MultiThreading MultiThreading)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})


if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(test)
    endif()
endif()
