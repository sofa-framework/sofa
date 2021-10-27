cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Compat LANGUAGES CXX)

set(SOFACOMPONENTCOMPATSRC_ROOT "src")

set(HEADER_FILES
    # ${SOFACOMPONENTCOMPATSRC_ROOT}/config.h.in 

)

set(SOURCE_FILES
    ${SOFACOMPONENTCOMPATSRC_ROOT}/init.cpp
)

include(Sofa.Component.ODESolver.Backward.cmake)
include(Sofa.Component.ODESolver.Forward.cmake)

sofa_find_package(Sofa.Config REQUIRED)


set(TARGET_NAME ${PROJECT_NAME})

add_library(${TARGET_NAME} INTERFACE ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} INTERFACE Sofa.Config)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(${TARGET_NAME} INTERFACE "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${TARGET_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src/")
