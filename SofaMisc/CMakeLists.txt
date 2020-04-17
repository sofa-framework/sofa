cmake_minimum_required(VERSION 3.1)
project(SofaMisc)

## Dependencies
find_package(SofaGeneral REQUIRED)

## Version
set(SOFAMISC_VERSION ${SofaFramework_VERSION})

## Subdirectories
set(SOFAMISC_TARGETS
    SofaMiscExtra
    SofaMiscEngine
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology
)

foreach(TARGET ${SOFAMISC_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaMisc) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAMISC_VERSION})
endforeach()

set(SOFAMISC_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAMISC_SRC}/config.h.in 
    ${SOFAMISC_SRC}/initSofaMisc.h
    )

set(SOURCE_FILES
    ${SOFAMISC_SRC}/initSofaMisc.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

foreach(target ${SOFAMISC_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAMISC")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  ${SOFAMISC_VERSION})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${SOFAMISC_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
    )
