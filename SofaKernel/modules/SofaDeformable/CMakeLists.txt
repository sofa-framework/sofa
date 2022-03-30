cmake_minimum_required(VERSION 3.12)
project(SofaDeformable LANGUAGES CXX)

set(SOFADEFORMABLE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADEFORMABLE_SRC}/config.h.in
    ${SOFADEFORMABLE_SRC}/initSofaDeformable.h
)

set(SOURCE_FILES
    ${SOFADEFORMABLE_SRC}/initSofaDeformable.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.SolidMechanics.Spring instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)

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
cmake_dependent_option(SOFADEFORMABLE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFADEFORMABLE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
