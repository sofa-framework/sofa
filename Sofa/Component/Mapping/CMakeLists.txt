cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Mapping LANGUAGES CXX)

set(SOFACOMPONENTMAPPING_SOURCE_DIR "src/sofa/component/mapping")

sofa_add_subdirectory_modules(SOFACOMPONENTMAPPING_TARGETS
    DIRECTORIES Linear NonLinear MappedMatrix
)

set(HEADER_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTMAPPING_TARGETS})

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
