cmake_minimum_required(VERSION 3.1)

project(SofaValidation_test)

find_package(SofaGeneral)
find_package(SofaCommon)
find_package(SofaBase)

set(SOURCE_FILES
  Monitor_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaValidation SofaLoader SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
