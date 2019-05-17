cmake_minimum_required(VERSION 3.1)
project(InvertibleFVM VERSION 0.1)


set(HEADER_FILES
    InvertibleFVMForceField.h
    InvertibleFVMForceField.inl
    config.h
)

set(SOURCE_FILES
    InvertibleFVMForceField.cpp
    initPlugin.cpp
)

set(README_FILES InvertibleFVM.txt)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_INVERTIBLEFVM")
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
