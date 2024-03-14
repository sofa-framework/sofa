cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Topology LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGY_SOURCE_DIR "src/sofa/component/topology")

sofa_add_subdirectory_modules(SOFACOMPONENTTOPOLOGY_TARGETS
    DIRECTORIES Container Mapping Utility
)

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTTOPOLOGY_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

if(SOFA_BUILD_TESTS)
    add_subdirectory(Testing)
endif()
