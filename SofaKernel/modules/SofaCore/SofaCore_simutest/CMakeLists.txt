cmake_minimum_required(VERSION 3.12)

project(SofaCore_simutest)

find_package(SofaBase REQUIRED)

set(SOURCE_FILES
    objectmodel/Base_test.cpp
    objectmodel/BaseContext_test.cpp
    objectmodel/BaseLink_simutest.cpp
    objectmodel/PathResolver_simutest.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaBaseUtils)

add_test(NAME SofaCore_simutest COMMAND SofaCore_simutest)
