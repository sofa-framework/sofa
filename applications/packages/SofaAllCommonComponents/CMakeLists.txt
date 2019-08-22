cmake_minimum_required(VERSION 3.1)
project(SofaAllCommonComponents VERSION 1.0)

set(HEADER_FILES
    config.h
    initAllCommonComponents.h
    )
set(SOURCE_FILES
    initAllCommonComponents.cpp
    )
set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFALLCOMMONCOMPONENTS -DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

find_package(SofaOpenglVisual QUIET)
if(SofaOpenglVisual_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME SofaAllCommonComponents
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "SofaAllCommonComponents/SofaAllCommonComponents"
    )
