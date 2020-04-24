cmake_minimum_required(VERSION 3.1)
project(SofaAdvanced)

## Dependencies
find_package(SofaGeneral REQUIRED)

## Version
set(SOFAADVANCED_VERSION ${SofaFramework_VERSION})

## Subdirectories
set(SOFAADVANCED_TARGETS
    SofaNonUniformFem
)

message(DEPRECATION "SofaAdvanced collection is deprecated, please directly use its modules in your project(s). It contains the following module(s): ${SOFAADVANCED_TARGETS}")

foreach(TARGET ${SOFAADVANCED_TARGETS})
    find_package(${TARGET} QUIET)
    if(NOT ${TARGET}_FOUND)
        sofa_force_add_outofree_module(../modules/${TARGET} ${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET} ON)
    endif()
endforeach()


set(SOFAADVANCED_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAADVANCED_SRC}/config.h.in 
    ${SOFAADVANCED_SRC}/initSofaAdvanced.h
    )

set(SOURCE_FILES
    ${SOFAADVANCED_SRC}/initSofaAdvanced.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

foreach(target ${SOFAADVANCED_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAADVANCED")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  ${SOFAADVANCED_VERSION})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${SOFAADVANCED_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
    )
