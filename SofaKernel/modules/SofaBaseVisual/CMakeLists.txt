cmake_minimum_required(VERSION 3.12)
project(SofaBaseVisual LANGUAGES CXX)

set(SOFABASEVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEVISUAL_SRC}/config.h.in
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.h
    ${SOFABASEVISUAL_SRC}/BackgroundSetting.h
)

set(SOURCE_FILES
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.cpp
    ${SOFABASEVISUAL_SRC}/BackgroundSetting.cpp
)

sofa_find_package(Sofa.SimulationCore REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Visual instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Visual REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Visual)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASEVISUAL_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASEVISUAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
