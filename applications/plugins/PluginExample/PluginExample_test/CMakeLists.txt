cmake_minimum_required(VERSION 3.1)

project(PluginExample_test)

find_package(PluginExample REQUIRED)

set(SOURCE_FILES
    MyBehaviorModel_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGTestMain PluginExample)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
