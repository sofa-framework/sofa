set(SOFACONSTRAINT_SRC src/SofaConstraint)
set(SOFAGENERALANIMATIONLOOP_SRC src/SofaGeneralAnimationLoop)

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/MappingGeometricStiffnessForceField.h
    ${SOFACONSTRAINT_SRC}/MappingGeometricStiffnessForceField.inl
    ${SOFAGENERALANIMATIONLOOP_SRC}/MechanicalMatrixMapper.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MechanicalMatrixMapper.inl
)