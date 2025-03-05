cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Diffusion LANGUAGES CXX)

set(SOFACOMPONENTDIFFUSION_SOURCE_DIR "src/sofa/component/diffusion")

set(HEADER_FILES
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/init.h
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/TetrahedronDiffusionFEMForceField.h
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/TetrahedronDiffusionFEMForceField.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTDIFFUSION_SOURCE_DIR}/TetrahedronDiffusionFEMForceField.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_DIFFUSION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_DIFFUSION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
