cmake_minimum_required(VERSION 3.12)

project(SofaGeneralDeformable_test)

set(SOURCE_FILES
    )

find_package(SofaGeneralDeformable REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaGeneralDeformable)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
