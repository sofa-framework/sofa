set(SOFANONUNIFORMFEM_SRC src/SofaNonUniformFem)

list(APPEND HEADER_FILES
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.inl
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.inl
    ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMForceFieldAndMass.inl
    ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMMapping.h
    ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMMapping.inl
)
