cmake_minimum_required(VERSION 3.12)
project(SofaEngine LANGUAGES CXX)

set(SOFAENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAENGINE_SRC}/config.h.in
    ${SOFAENGINE_SRC}/initSofaEngine.h
)

set(SOURCE_FILES
    ${SOFAENGINE_SRC}/initSofaEngine.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Select instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Engine.Select REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Select)

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
cmake_dependent_option(SOFAENGINE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
