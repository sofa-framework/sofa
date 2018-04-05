cmake_minimum_required(VERSION 3.1)

project(SofaMisc_test)

set(HEADER_FILES)
set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    AddResourceRepository_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DMISC_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
