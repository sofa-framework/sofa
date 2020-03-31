cmake_minimum_required(VERSION 3.1)
project(SofaMiscTopology VERSION 1.0 LANGUAGES CXX)

find_package(SofaBase REQUIRED)
sofa_find_package(ZLIB BOTH_SCOPES)

set(SOFAMISCTOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
	${SOFAMISCTOPOLOGY_SRC}/config.h.in
	${SOFAMISCTOPOLOGY_SRC}/initMiscTopology.h
)

set(SOURCE_FILES
	${SOFAMISCTOPOLOGY_SRC}/initMiscTopology.cpp
)

list(APPEND HEADER_FILES
	${SOFAMISCTOPOLOGY_SRC}/TopologicalChangeProcessor.h
)
list(APPEND SOURCE_FILES
	${SOFAMISCTOPOLOGY_SRC}/TopologicalChangeProcessor.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
if(ZLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS  "-DSOFA_BUILD_MISC_TOPOLOGY")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/${PROJECT_NAME}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCTOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCTOPOLOGY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscTopology_test)
endif()

