cmake_minimum_required(VERSION 3.12)

project(SofaBaseUtils_test)

set(SOURCE_FILES
        MessageHandlerComponent_test.cpp
        MakeAliasComponent_test.cpp
        MakeDataAliasComponent_test.cpp
        RequiredPlugin_test.cpp
        AddResourceRepository_test.cpp
)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#find_package(SofaTest REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)
add_definitions("-DSOFABASEUTILS_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
