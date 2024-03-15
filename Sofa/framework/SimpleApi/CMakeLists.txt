cmake_minimum_required(VERSION 3.12)
project(Sofa.SimpleApi LANGUAGES CXX)

set(SOFASIMPLEAPI_SRC "src/sofa/simpleapi")

set(HEADER_FILES
    ${SOFASIMPLEAPI_SRC}/config.h.in
    ${SOFASIMPLEAPI_SRC}/init.h
    ${SOFASIMPLEAPI_SRC}/SimpleApi.h
)

set(SOURCE_FILES
    ${SOFASIMPLEAPI_SRC}/init.cpp
    ${SOFASIMPLEAPI_SRC}/SimpleApi.cpp
)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Graph)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_SIMPLEAPI_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_SIMPLEAPI_BUILD_TESTS)
    add_subdirectory(test)
endif()
