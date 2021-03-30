cmake_minimum_required(VERSION 3.12)
project(SofaMiscFem LANGUAGES CXX)

set(SOFAMISCFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFEM_SRC}/config.h.in
    ${SOFAMISCFEM_SRC}/initSofaMiscFem.h
    )

set(SOURCE_FILES
    ${SOFAMISCFEM_SRC}/initSofaMiscFem.cpp
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
	    ${SOFAMISCFEM_SRC}/QuadBendingFEMForceField.h
	    ${SOFAMISCFEM_SRC}/QuadBendingFEMForceField.inl
        )
    list(APPEND SOURCE_FILES
        ${SOFAMISCFEM_SRC}/PlasticMaterial.cpp
        ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.cpp
        ${SOFAMISCFEM_SRC}/TriangularFEMForceField.cpp
	    ${SOFAMISCFEM_SRC}/QuadBendingFEMForceField.cpp
        )
endif()

# Dependencies
sofa_find_package(Newmat BOTH_SCOPES)
sofa_find_package(SofaBase REQUIRED) 
sofa_find_package(SofaSimpleFem REQUIRED) 

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaBaseMechanics)
if(Newmat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC newmat)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )


sofa_configure_unity_build(TARGET ${PROJECT_NAME}
                           ACTIVATED "${SOFA_BUILD_WITH_UNITY_BUILD_ENABLED}"
                           BATCHSIZE "${SOFA_UNITY_BUILD_BATCH_SIZE}")

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCFEM_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCFEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscFem_test)
endif()
