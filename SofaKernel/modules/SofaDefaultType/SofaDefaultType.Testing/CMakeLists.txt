cmake_minimum_required(VERSION 3.12)
project(SofaDefaultType.Testing LANGUAGES CXX)

set(HEADER_FILES
    src/SofaDefaultType.Testing/BaseMatrix_test.h
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")