set(SOFAMISCFEM_SRC src/SofaMiscFem)

list(APPEND HEADER_FILES
    ${SOFAMISCFEM_SRC}/BaseMaterial.h
    ${SOFAMISCFEM_SRC}/BoyceAndArruda.h
    ${SOFAMISCFEM_SRC}/Costa.h
    ${SOFAMISCFEM_SRC}/HyperelasticMaterial.h
    ${SOFAMISCFEM_SRC}/MooneyRivlin.h
    ${SOFAMISCFEM_SRC}/NeoHookean.h
    ${SOFAMISCFEM_SRC}/Ogden.h
    ${SOFAMISCFEM_SRC}/PlasticMaterial.h
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.h
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.inl
    ${SOFAMISCFEM_SRC}/STVenantKirchhoff.h
    ${SOFAMISCFEM_SRC}/TetrahedronHyperelasticityFEMForceField.h
    ${SOFAMISCFEM_SRC}/TetrahedronHyperelasticityFEMForceField.inl
    ${SOFAMISCFEM_SRC}/VerondaWestman.h
)
