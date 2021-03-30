cmake_minimum_required(VERSION 3.12)
project(SofaDeformable LANGUAGES CXX)

set(SOFADEFORMABLE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADEFORMABLE_SRC}/config.h.in
    ${SOFADEFORMABLE_SRC}/initSofaDeformable.h
    ${SOFADEFORMABLE_SRC}/AngularSpringForceField.h
    ${SOFADEFORMABLE_SRC}/AngularSpringForceField.inl
    ${SOFADEFORMABLE_SRC}/MeshSpringForceField.h
    ${SOFADEFORMABLE_SRC}/MeshSpringForceField.inl
    ${SOFADEFORMABLE_SRC}/RestShapeSpringsForceField.h
    ${SOFADEFORMABLE_SRC}/RestShapeSpringsForceField.inl
    ${SOFADEFORMABLE_SRC}/PolynomialRestShapeSpringsForceField.h
    ${SOFADEFORMABLE_SRC}/PolynomialRestShapeSpringsForceField.inl
    ${SOFADEFORMABLE_SRC}/SpringForceField.h
    ${SOFADEFORMABLE_SRC}/SpringForceField.inl
    ${SOFADEFORMABLE_SRC}/StiffSpringForceField.h
    ${SOFADEFORMABLE_SRC}/StiffSpringForceField.inl
    ${SOFADEFORMABLE_SRC}/PolynomialSpringsForceField.h
    ${SOFADEFORMABLE_SRC}/PolynomialSpringsForceField.inl
)

set(SOURCE_FILES
    ${SOFADEFORMABLE_SRC}/initSofaDeformable.cpp
    ${SOFADEFORMABLE_SRC}/AngularSpringForceField.cpp
    ${SOFADEFORMABLE_SRC}/MeshSpringForceField.cpp
    ${SOFADEFORMABLE_SRC}/RestShapeSpringsForceField.cpp
    ${SOFADEFORMABLE_SRC}/PolynomialRestShapeSpringsForceField.cpp
    ${SOFADEFORMABLE_SRC}/SpringForceField.cpp
    ${SOFADEFORMABLE_SRC}/StiffSpringForceField.cpp
    ${SOFADEFORMABLE_SRC}/PolynomialSpringsForceField.cpp
)

sofa_find_package(SofaBase REQUIRED) # SofaBaseTopology SofaEigen2Solver

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaEigen2Solver)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_configure_unity_build(TARGET ${PROJECT_NAME}
                           ACTIVATED "${SOFA_BUILD_WITH_UNITY_BUILD_ENABLED}"
                           BATCHSIZE "${SOFA_UNITY_BUILD_BATCH_SIZE}")

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
cmake_dependent_option(SOFADEFORMABLE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFADEFORMABLE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
