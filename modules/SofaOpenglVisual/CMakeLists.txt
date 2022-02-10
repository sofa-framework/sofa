cmake_minimum_required(VERSION 3.12)
project(SofaOpenglVisual LANGUAGES CXX)

sofa_find_package(SofaFramework REQUIRED) # SofaSimulationCore
sofa_find_package(SofaBaseVisual REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)

set(HEADER_FILES
    src/SofaOpenglVisual/config.h
    src/SofaOpenglVisual/initSofaOpenglVisual.h
    )
set(SOURCE_FILES
    src/SofaOpenglVisual/initSofaOpenglVisual.cpp
    )
set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual SofaSimulationCore Sofa.GL)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaOpenglVisual"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAOPENGLVISUAL_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAOPENGLVISUAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaOpenglVisual_test)
endif()
