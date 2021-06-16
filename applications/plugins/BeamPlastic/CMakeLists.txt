cmake_minimum_required(VERSION 3.1)
project(BeamPlastic VERSION 1.0)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    config.h
    forceField/BeamPlasticFEMForceField.h
    forceField/BeamPlasticFEMForceField.inl
    constitutiveLaw/PlasticConstitutiveLaw.h
    constitutiveLaw/RambergOsgood.h
    quadrature/gaussian.h
    quadrature/quadrature.h
)

set(SOURCE_FILES
    initBeamPlastic.cpp
    forceField/BeamPlasticFEMForceField.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BEAMPLASTIC"
	                                             DEBUG_POSTFIX "_d")

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

target_link_libraries(${PROJECT_NAME} SofaCore SofaBaseTopology SofaSimpleFem SofaGeneralSimpleFem SofaEigen2Solver Sofa.GL)

sofa_create_package_with_targets(
    PACKAGE_NAME BeamPlastic
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "BeamPlastic"
    EXAMPLE_INSTALL_DIR "examples"
    RELOCATABLE "plugins"
    )

option(BEAMPLASTIC_BUILD_TESTS "Build unit tests" ON)
if(BEAMPLASTIC_BUILD_TESTS)
    add_subdirectory(BeamPlastic_test)
endif()

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET_PROPERTY(TARGET ${PROJECT_NAME} PROPERTY FOLDER "plugins")