cmake_minimum_required(VERSION 3.12)
project(SofaMiscTopology LANGUAGES CXX)

set(SOFAMISCTOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
	${SOFAMISCTOPOLOGY_SRC}/config.h.in
	${SOFAMISCTOPOLOGY_SRC}/initSofaMiscTopology.h
)

set(SOURCE_FILES
	${SOFAMISCTOPOLOGY_SRC}/initSofaMiscTopology.cpp
)

list(APPEND HEADER_FILES
	${SOFAMISCTOPOLOGY_SRC}/TopologicalChangeProcessor.h
	${SOFAMISCTOPOLOGY_SRC}/TopologyChecker.h
)
list(APPEND SOURCE_FILES
	${SOFAMISCTOPOLOGY_SRC}/TopologicalChangeProcessor.cpp
	${SOFAMISCTOPOLOGY_SRC}/TopologyChecker.cpp
)

sofa_find_package(SofaBase REQUIRED) # SofaBaseTopology
sofa_find_package(ZLIB BOTH_SCOPES)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
if(ZLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
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
