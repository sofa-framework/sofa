set(SOFADEFORMABLE_SRC src/SofaDeformable)
set(SOFARIGID_SRC src/SofaRigid)
set(SOFAGENERALDEFORMABLE_SRC src/SofaGeneralDeformable)
set(SOFAGENERALOBJECTINTERACTION_SRC src/SofaGeneralObjectInteraction)
set(SOFAMISCFORCEFIELD_SRC src/SofaMiscForceField)

list(APPEND HEADER_FILES
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
    ${SOFARIGID_SRC}/JointSpring.h
    ${SOFARIGID_SRC}/JointSpring.inl
    ${SOFARIGID_SRC}/JointSpringForceField.h
    ${SOFARIGID_SRC}/JointSpringForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/FastTriangularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/FastTriangularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/FrameSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/FrameSpringForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/QuadBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/QuadBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/QuadularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/QuadularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/RegularGridSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/RegularGridSpringForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangleBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangleBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBiquadraticSpringsForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBiquadraticSpringsForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularQuadraticSpringsForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularQuadraticSpringsForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/VectorSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/VectorSpringForceField.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/RepulsiveSpringForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/RepulsiveSpringForceField.inl
    ${SOFAMISCFORCEFIELD_SRC}/GearSpringForceField.h
    ${SOFAMISCFORCEFIELD_SRC}/GearSpringForceField.inl
)
