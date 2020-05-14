cmake_minimum_required(VERSION 3.1)

project(SofaCore_simutest)

find_package(SofaBase REQUIRED)
find_package(SofaComponentAll REQUIRED)

set(SOURCE_FILES
    objectmodel/Base_test.cpp
    objectmodel/BaseContext_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaBase)

add_test(NAME SofaCore_simutest COMMAND SofaCore_simutest)
