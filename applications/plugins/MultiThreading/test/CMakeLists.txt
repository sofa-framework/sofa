cmake_minimum_required(VERSION 3.1)

project(MultiThreading_test)

set ( HEADER_FILES
	TaskSchedulerTestTasks.h
)
set(SOURCE_FILES
        TaskSchedulerTests.cpp
		TaskSchedulerTestTasks.cpp
)

find_package(SofaTest REQUIRED)
find_package(MultiThreading REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
#add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain MultiThreading)
target_include_directories(${PROJECT_NAME} PRIVATE ${MultiThreading_INCLUDE_DIRS})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})