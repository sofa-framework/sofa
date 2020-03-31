cmake_minimum_required(VERSION 3.1)
project(SofaMiscFem VERSION 1.0 LANGUAGES CXX)

# Dependencies
sofa_find_package(Newmat BOTH_SCOPES)
find_package(SofaCommon REQUIRED)

set(SOFAMISCFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFEM_SRC}/config.h.in
    ${SOFAMISCFEM_SRC}/initMiscFEM.h
    )

set(SOURCE_FILES
    ${SOFAMISCFEM_SRC}/initMiscFEM.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCFEM_SRC}/BaseMaterial.h
    ${SOFAMISCFEM_SRC}/BoyceAndArruda.h
    ${SOFAMISCFEM_SRC}/Costa.h
    ${SOFAMISCFEM_SRC}/FastTetrahedralCorotationalForceField.h
    ${SOFAMISCFEM_SRC}/FastTetrahedralCorotationalForceField.inl
    ${SOFAMISCFEM_SRC}/HyperelasticMaterial.h
    ${SOFAMISCFEM_SRC}/MooneyRivlin.h
    ${SOFAMISCFEM_SRC}/NeoHookean.h
    ${SOFAMISCFEM_SRC}/Ogden.h
    ${SOFAMISCFEM_SRC}/STVenantKirchhoff.h
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.h
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.inl
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.h
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.inl
    ${SOFAMISCFEM_SRC}/VerondaWestman.h

    ${SOFAMISCFEM_SRC}/TetrahedronHyperelasticityFEMForceField.h
    ${SOFAMISCFEM_SRC}/TetrahedronHyperelasticityFEMForceField.inl
    ${SOFAMISCFEM_SRC}/TriangleFEMForceField.h
    ${SOFAMISCFEM_SRC}/TriangleFEMForceField.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCFEM_SRC}/FastTetrahedralCorotationalForceField.cpp
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.cpp
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.cpp

    ${SOFAMISCFEM_SRC}/TetrahedronHyperelasticityFEMForceField.cpp
    ${SOFAMISCFEM_SRC}/TriangleFEMForceField.cpp
    )

if(Newmat_FOUND)
    list(APPEND HEADER_FILES
        ${SOFAMISCFEM_SRC}/PlasticMaterial.h
        ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.h
        ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.inl
        ${SOFAMISCFEM_SRC}/TriangularFEMForceField.h
        ${SOFAMISCFEM_SRC}/TriangularFEMForceField.inl
        )
    list(APPEND SOURCE_FILES
        ${SOFAMISCFEM_SRC}/PlasticMaterial.cpp
        ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.cpp
        ${SOFAMISCFEM_SRC}/TriangularFEMForceField.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem)
if(Newmat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC newmat)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/${PROJECT_NAME}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCFEM_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCFEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscFem_test)
endif()

