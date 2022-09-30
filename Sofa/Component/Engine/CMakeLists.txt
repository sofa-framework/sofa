cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Engine LANGUAGES CXX)

set(SOFACOMPONENTENGINE_SOURCE_DIR "src/sofa/component/engine")

sofa_add_subdirectory_modules(SOFACOMPONENTENGINE_TARGETS
    DIRECTORIES Analyze Generate Select Transform
)

set(HEADER_FILES
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTENGINE_TARGETS})

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
