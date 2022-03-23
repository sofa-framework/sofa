cmake_minimum_required(VERSION 3.12)
project(SofaBaseMechanics LANGUAGES CXX)

set(SOFABASEMECHANICS_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEMECHANICS_SRC}/config.h.in
    ${SOFABASEMECHANICS_SRC}/initSofaBaseMechanics.h
)

set(SOURCE_FILES
    ${SOFABASEMECHANICS_SRC}/initSofaBaseMechanics.cpp

)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Mass, Sofa.Component.Mapping.Linear and Sofa.Component.StateContainer instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mass REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mass)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.Linear)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.StateContainer)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASEMECHANICS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASEMECHANICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
