cmake_minimum_required(VERSION 2.8)
project(SofaGuiGlut)

set(HEADER_FILES
	SimpleGUI.h)

set(SOURCE_FILES
	SimpleGUI.cpp)

if(SOFA_HAVE_BOOST)
	list(APPEND HEADER_FILES MultithreadGUI.h)

	list(APPEND SOURCE_FILES MultithreadGUI.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaGuiCommon)

set(SOFAGUIGLUT_COMPILER_FLAGS "-DSOFA_BUILD_SOFAGUIGLUT")

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAGUI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAGUIGLUT_COMPILER_FLAGS}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/SofaGui")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_DEPENDENCY_DIR}>")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT SofaGuiTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${SofaGuiGlut_SOURCE_DIR}/ DESTINATION "include/SofaGuiGlut"
        COMPONENT SofaGui_headers
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp")
