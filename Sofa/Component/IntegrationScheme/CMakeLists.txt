cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.IntegrationScheme LANGUAGES CXX)

set(SOFACOMPONENTINTEGRATIONSCHEME_SOURCE_DIR "src/sofa/component/integrationscheme")

sofa_add_subdirectory_modules(SOFACOMPONENTINTEGRATIONSCHEME_TARGETS
    DIRECTORIES Forward Backward
)

set(HEADER_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEME_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTINTEGRATIONSCHEME_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEME_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTINTEGRATIONSCHEME_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# CMake-level compat: redirect find_package(Sofa.Component.ODESolver) with an error
configure_package_config_file(
    compat/Sofa.Component.ODESolverConfig.cmake.in
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolverConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/Sofa.Component.ODESolver"
)
install(FILES
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolverConfig.cmake"
    DESTINATION "lib/cmake/Sofa.Component.ODESolver"
    COMPONENT headers
)

