set(SOFASIMPLEFEM_SRC src/SofaSimpleFem)
set(SOFAGENERALDEFORMABLE_SRC src/SofaGeneralDeformable)

list(APPEND HEADER_FILES
    ${SOFASIMPLEFEM_SRC}/fwd.h
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.h
    ${SOFAMISCFEM_SRC}/TetrahedralTensorMassForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularTensorMassForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularTensorMassForceField.inl
)
