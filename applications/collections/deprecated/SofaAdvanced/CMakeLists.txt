cmake_minimum_required(VERSION 3.12)
project(SofaAdvanced)

## Version
set(SOFAADVANCED_VERSION ${SofaFramework_VERSION})

## Subdirectories
set(SOFAADVANCED_MODULES
    SofaNonUniformFem
    )

foreach(module ${SOFAADVANCED_MODULES})
    sofa_find_package(${module} QUIET)
    if(NOT ${module}_FOUND)
        string(TOUPPER module_${module} MODULE_TO_CHECK)
        list(APPEND LIST_MODULES_TO_CHECK ${MODULE_TO_CHECK})
    endif()
endforeach()
if(LIST_MODULES_TO_CHECK)
    message(FATAL_ERROR
        " If you still want to use ${PROJECT_NAME}, please enable the following CMake option(s): \n"
        " ${LIST_MODULES_TO_CHECK}"
        )
endif()

set(SOFAADVANCED_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAADVANCED_SRC}/config.h.in 
    ${SOFAADVANCED_SRC}/initSofaAdvanced.h
    )

set(SOURCE_FILES
    ${SOFAADVANCED_SRC}/initSofaAdvanced.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAADVANCED_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
