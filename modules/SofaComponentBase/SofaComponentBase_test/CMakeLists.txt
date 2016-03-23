cmake_minimum_required(VERSION 2.8.12)

project(SofaComponentBase_test)

set(SOURCE_FILES
        MessageHandlerComponent_test.cpp       
)

find_package(SofaTest REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
