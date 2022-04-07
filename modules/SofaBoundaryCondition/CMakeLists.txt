cmake_minimum_required(VERSION 3.12)
project(SofaBoundaryCondition LANGUAGES CXX)

set(SOFABOUNDARYCONDITION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/config.h.in
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.h
    )

set(SOURCE_FILES
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.cpp
    )

list(APPEND HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.inl
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.h
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.inl
    )

list(APPEND SOURCE_FILES
    ${SOFABOUNDARYCONDITION_SRC}/ConicalForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/ConstantForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/DiagonalVelocityDampingForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/EdgePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/EllipsoidForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/LinearForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/OscillatingTorsionPressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/PlaneForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/QuadPressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/SphereForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/SurfacePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TaitSurfacePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TorsionForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/TrianglePressureForceField.cpp
    ${SOFABOUNDARYCONDITION_SRC}/UniformVelocityDampingForceField.cpp
    )
    
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaBaseLinearSolver REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseLinearSolver)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Constraint.Projective instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Projective)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABOUNDARYCONDITION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABOUNDARYCONDITION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
