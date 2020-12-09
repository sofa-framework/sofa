cmake_minimum_required(VERSION 3.12)

project(SofaGeneralSimpleFem_test)

set(SOURCE_FILES ../../empty.cpp)

find_package(SofaGeneralSimpleFem REQUIRED)

list(APPEND SOURCE_FILES
)

add_definitions("-DSOFAGENERALSIMPLEFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaGeneralSimpleFem)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
