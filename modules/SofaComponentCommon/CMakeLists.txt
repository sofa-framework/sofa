cmake_minimum_required(VERSION 2.8)
project(SofaComponentCommon)

set(HEADER_FILES

    initComponentCommon.h

    )

set(SOURCE_FILES

    initComponentCommon.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(SOFACOMPONENTCOMMON_DEPS ${SOFACOMMON_TARGETS})
list(REMOVE_ITEM SOFACOMPONENTCOMMON_DEPS SofaComponentCommon)

target_link_libraries( ${PROJECT_NAME} ${SOFACOMPONENTCOMMON_DEPS})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFACOMPONENTCOMMON_COMPILER_FLAGS "-DSOFA_BUILD_COMPONENT_COMMON")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACOMPONENTCOMMON_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaCommon_libraries
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        )
        
install(DIRECTORY ${SofaComponentCommon_SOURCE_DIR}/ DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/SofaComponentCommon"
        COMPONENT SofaCommon_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
