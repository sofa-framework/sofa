cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Mass LANGUAGES CXX)

set(SOFACOMPONENTMASS_SOURCE_DIR "src/sofa/component/mass")

set(HEADER_FILES
    ${SOFACOMPONENTMASS_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMASS_SOURCE_DIR}/init.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/AddMToMatrixFunctor.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/DiagonalMass.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/MassType.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/MeshMatrixMass.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/MeshMatrixMass.inl
    ${SOFACOMPONENTMASS_SOURCE_DIR}/RigidMassType.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/UniformMass.h
    ${SOFACOMPONENTMASS_SOURCE_DIR}/UniformMass.inl
    ${SOFACOMPONENTMASS_SOURCE_DIR}/VecMassType.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMASS_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTMASS_SOURCE_DIR}/DiagonalMass.cpp
    ${SOFACOMPONENTMASS_SOURCE_DIR}/MeshMatrixMass.cpp
    ${SOFACOMPONENTMASS_SOURCE_DIR}/UniformMass.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Dynamic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_MASS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_MASS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
