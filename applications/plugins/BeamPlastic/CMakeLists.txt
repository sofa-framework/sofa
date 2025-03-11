cmake_minimum_required(VERSION 3.1)
project(BeamPlastic VERSION 1.0)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Core REQUIRED)


set(BEAMPLASTIC_SRC src/BeamPlastic)
set(HEADER_FILES
    ${BEAMPLASTIC_SRC}/config.h.in
    ${BEAMPLASTIC_SRC}/init.h
    ${BEAMPLASTIC_SRC}/forcefield/BeamPlasticFEMForceField.h
    ${BEAMPLASTIC_SRC}/forcefield/BeamPlasticFEMForceField.inl
    ${BEAMPLASTIC_SRC}/constitutivelaw/PlasticConstitutiveLaw.h
    ${BEAMPLASTIC_SRC}/constitutivelaw/RambergOsgood.h
    ${BEAMPLASTIC_SRC}/quadrature/gaussian.h
    ${BEAMPLASTIC_SRC}/quadrature/quadrature.h
)

set(SOURCE_FILES
    ${BEAMPLASTIC_SRC}/init.cpp
    ${BEAMPLASTIC_SRC}/forcefield/BeamPlasticFEMForceField.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    EXAMPLE_INSTALL_DIR "examples"
    RELOCATABLE "plugins"
    )

option(BEAMPLASTIC_BUILD_TESTS "Build unit tests" ON)
if(BEAMPLASTIC_BUILD_TESTS)
    add_subdirectory(BeamPlastic_test)
endif()
