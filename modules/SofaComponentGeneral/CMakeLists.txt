cmake_minimum_required(VERSION 2.8.11)
project(SofaComponentGeneral)

set(HEADER_FILES

    initComponentGeneral.h
    )

set(SOURCE_FILES

    initComponentGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(SOFACOMPONENTGENERAL_DEPS ${SOFAGENERAL_TARGETS})
list(REMOVE_ITEM SOFACOMPONENTGENERAL_DEPS SofaComponentGeneral)

target_link_libraries(${PROJECT_NAME} ${SOFACOMPONENTGENERAL_DEPS})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFACOMPONENTGENERAL_COMPILER_FLAGS "-DSOFA_BUILD_COMPONENT_GENERAL")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACOMPONENTGENERAL_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGeneral_libraries
        EXPORT  SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        )
        
install(DIRECTORY ${SofaComponentGeneral_SOURCE_DIR}/ DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/SofaComponentGeneral"
        COMPONENT SofaGeneral_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
