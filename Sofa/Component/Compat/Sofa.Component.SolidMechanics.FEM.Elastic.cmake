set(SOFASIMPLEFEM_SRC src/SofaSimpleFem)
set(SOFAGENERALSIMPLEFEM_SRC src/SofaGeneralSimpleFem)
set(SOFAMISCFEM_SRC src/SofaMiscFem)

list(APPEND HEADER_FILES
    ${SOFASIMPLEFEM_SRC}/fwd.h
    ${SOFASIMPLEFEM_SRC}/HexahedronFEMForceField.h
    ${SOFASIMPLEFEM_SRC}/HexahedronFEMForceField.inl
    ${SOFASIMPLEFEM_SRC}/TetrahedronFEMForceField.h
    ${SOFASIMPLEFEM_SRC}/TetrahedronFEMForceField.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/BeamFEMForceField.h
    ${SOFAGENERALSIMPLEFEM_SRC}/BeamFEMForceField.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedralFEMForceField.h
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedralFEMForceField.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedralFEMForceFieldAndMass.h
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedralFEMForceFieldAndMass.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedronFEMForceFieldAndMass.h
    ${SOFAGENERALSIMPLEFEM_SRC}/HexahedronFEMForceFieldAndMass.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/TetrahedralCorotationalFEMForceField.h
    ${SOFAGENERALSIMPLEFEM_SRC}/TetrahedralCorotationalFEMForceField.inl
    ${SOFAGENERALSIMPLEFEM_SRC}/TriangularFEMForceFieldOptim.h
    ${SOFAGENERALSIMPLEFEM_SRC}/TriangularFEMForceFieldOptim.inl
    ${SOFAMISCFEM_SRC}/FastTetrahedralCorotationalForceField.h
    ${SOFAMISCFEM_SRC}/FastTetrahedralCorotationalForceField.inl
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.h
    ${SOFAMISCFEM_SRC}/StandardTetrahedralFEMForceField.inl
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.h
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.inl
    ${SOFAMISCFEM_SRC}/TriangleFEMForceField.h
    ${SOFAMISCFEM_SRC}/TriangleFEMForceField.inl
    ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.h
    ${SOFAMISCFEM_SRC}/TriangularAnisotropicFEMForceField.inl
    ${SOFAMISCFEM_SRC}/TriangularFEMForceField.h
    ${SOFAMISCFEM_SRC}/TriangularFEMForceField.inl
    ${SOFAMISCFEM_SRC}/QuadBendingFEMForceField.h
    ${SOFAMISCFEM_SRC}/QuadBendingFEMForceField.inl
)
