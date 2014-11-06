cmake_minimum_required(VERSION 2.8)
project(SofaGuiMain)

set(HEADER_FILES

	../Main.h
	)

set(SOURCE_FILES

	../Main.cpp
	)

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(SOFAGUIMAIN_DEPS ${SOFAGUI_TARGETS})
list(REMOVE_ITEM SOFAGUIMAIN_DEPS SofaGuiMain)

target_link_libraries( ${PROJECT_NAME} ${SOFAGUIMAIN_DEPS})

set( SOFAGUIMAIN_COMPILER_FLAGS "-DSOFA_BUILD_GUIMAIN" )

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGUI_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAGUIMAIN_COMPILER_FLAGS}"  )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_DEPENDENCY_DIR}>")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT  SofaGuiTargets
        RUNTIME DESTINATION "${SOFAGUI_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        )

install(DIRECTORY ${SofaGuiMain_SOURCE_DIR}/ DESTINATION "${SOFAGUI_INSTALL_INCLUDE_DIR}/SofaGuiMain"
        COMPONENT SofaGui_headers
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
