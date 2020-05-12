cmake_minimum_required(VERSION 3.1)
project(SofaComponentAll VERSION 1.0)

find_package(SofaBase REQUIRED)
find_package(SofaCommon REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)

set(HEADER_FILES
    config.h
    initComponentAll.h
    )
set(SOURCE_FILES
    initComponentAll.cpp
    )
set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaBase SofaCommon SofaGeneral SofaMisc)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACOMPONENTALL -DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

find_package(SofaOpenglVisual QUIET)
if(SofaOpenglVisual_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)
endif()

# TODO: remove SofaAllCommonComponents backward compatibility at SOFA v20.06
add_library(SofaAllCommonComponents SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(SofaAllCommonComponents ${PROJECT_NAME})
set_target_properties(SofaAllCommonComponents PROPERTIES COMPILE_FLAGS "-DSOFACOMPONENTALL_USING_DEPRECATED_NAME")

## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME SofaComponentAll
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} SofaAllCommonComponents
    INCLUDE_INSTALL_DIR "SofaComponentAll/SofaComponentAll"
    )
