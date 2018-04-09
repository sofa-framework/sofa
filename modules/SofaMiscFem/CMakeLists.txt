cmake_minimum_required(VERSION 3.1)
project(SofaMiscFem)

# Dependencies
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND})

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

if(SOFA_HAVE_NEWMAT)
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

if(NOT SOFA_NO_OPENGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()

if(SOFA_HAVE_NEWMAT)
    target_link_libraries(${PROJECT_NAME} PUBLIC newmat)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
