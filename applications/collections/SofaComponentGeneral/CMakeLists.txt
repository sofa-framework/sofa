cmake_minimum_required(VERSION 3.1)
project(SofaComponentGeneral VERSION 1.0)

find_package(SofaGeneral REQUIRED)

set(HEADER_FILES
    config.h
    initComponentGeneral.h
    )

set(SOURCE_FILES
    initComponentGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
foreach(target ${SOFAGENERAL_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_GENERAL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
    )
