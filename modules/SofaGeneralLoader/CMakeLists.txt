cmake_minimum_required(VERSION 3.12)
project(SofaGeneralLoader LANGUAGES CXX)

set(SOFAGENERALLOADER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALLOADER_SRC}/config.h.in
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.h
    )

set(SOURCE_FILES
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated; its components are in Sofa.Component.IO.Mesh and Sofa.Component.IO.Misc.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.IO.Mesh REQUIRED)
sofa_find_package(Sofa.Component.IO.Misc REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.IO.Mesh)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.IO.Misc)

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
cmake_dependent_option(SOFAGENERALLOADER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALLOADER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
