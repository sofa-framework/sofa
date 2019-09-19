cmake_minimum_required(VERSION 3.1)
project(SofaComponentAll VERSION 1.0)

find_package(SofaComponentBase REQUIRED)
find_package(SofaComponentCommon REQUIRED)
find_package(SofaComponentGeneral REQUIRED)
find_package(SofaComponentAdvanced REQUIRED)
find_package(SofaComponentMisc REQUIRED)

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
target_link_libraries(${PROJECT_NAME} SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACOMPONENTALL -DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")

find_package(SofaOpenglVisual QUIET)
if(SofaOpenglVisual_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME SofaComponentAll
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "SofaComponentAll/SofaComponentAll"
    )
