cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.MechanicalLoad LANGUAGES CXX)

set(SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR "src/sofa/component/mechanicalload")

set(HEADER_FILES
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/init.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConicalForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConicalForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConstantForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConstantForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/DiagonalVelocityDampingForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/DiagonalVelocityDampingForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EdgePressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EdgePressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EllipsoidForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EllipsoidForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/Gravity.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/InteractionEllipsoidForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/InteractionEllipsoidForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/LinearForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/LinearForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/OscillatingTorsionPressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/OscillatingTorsionPressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/PlaneForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/PlaneForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/QuadPressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/QuadPressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SphereForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SphereForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SurfacePressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SurfacePressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TaitSurfacePressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TaitSurfacePressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TorsionForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TorsionForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TrianglePressureForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TrianglePressureForceField.inl
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/UniformVelocityDampingForceField.h
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/UniformVelocityDampingForceField.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConicalForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/ConstantForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/DiagonalVelocityDampingForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EdgePressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/EllipsoidForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/Gravity.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/InteractionEllipsoidForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/LinearForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/OscillatingTorsionPressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/PlaneForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/QuadPressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SphereForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/SurfacePressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TaitSurfacePressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TorsionForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/TrianglePressureForceField.cpp
    ${SOFACOMPONENTMECHANICALLOAD_SOURCE_DIR}/UniformVelocityDampingForceField.cpp
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
cmake_dependent_option(SOFA_COMPONENT_MECHANICALLOAD_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_MECHANICALLOAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
