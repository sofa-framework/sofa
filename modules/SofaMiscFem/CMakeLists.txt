cmake_minimum_required(VERSION 3.12)
project(SofaMiscFem)

# Dependencies
sofa_find_package(Newmat BOTH_SCOPES)
find_package(SofaCommon REQUIRED) # SofaSimpleFem

set(HEADER_FILES
    config.h
    initMiscFEM.h
    )

set(SOURCE_FILES
    initMiscFEM.cpp
    )

list(APPEND HEADER_FILES
    BaseMaterial.h
    BoyceAndArruda.h
    Costa.h
    FastTetrahedralCorotationalForceField.h
    FastTetrahedralCorotationalForceField.inl
    HyperelasticMaterial.h
    MooneyRivlin.h
    NeoHookean.h
    Ogden.h
    STVenantKirchhoff.h
    StandardTetrahedralFEMForceField.h
    StandardTetrahedralFEMForceField.inl
    TetrahedralTensorMassForceField.h
    TetrahedralTensorMassForceField.inl
    VerondaWestman.h

    TetrahedronHyperelasticityFEMForceField.h
    TetrahedronHyperelasticityFEMForceField.inl
    TriangleFEMForceField.h
    TriangleFEMForceField.inl
    )
list(APPEND SOURCE_FILES
    FastTetrahedralCorotationalForceField.cpp
    StandardTetrahedralFEMForceField.cpp
    TetrahedralTensorMassForceField.cpp

    TetrahedronHyperelasticityFEMForceField.cpp
    TriangleFEMForceField.cpp
    )

if(Newmat_FOUND)
    list(APPEND HEADER_FILES
        PlasticMaterial.h
        TriangularAnisotropicFEMForceField.h
        TriangularAnisotropicFEMForceField.inl
        TriangularFEMForceField.h
        TriangularFEMForceField.inl
        )
    list(APPEND SOURCE_FILES
        PlasticMaterial.cpp
        TriangularAnisotropicFEMForceField.cpp
        TriangularFEMForceField.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem)
if(Newmat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC newmat)
endif()

sofa_add_targets_to_package(
    PACKAGE_NAME SofaMisc
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
