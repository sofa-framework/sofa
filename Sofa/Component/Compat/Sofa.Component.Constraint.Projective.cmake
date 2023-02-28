set(SOFAGENERALOBJECTINTERACTION_SRC src/SofaGeneralObjectInteraction)
set(SOFABOUNDARYCONDITION_SRC src/SofaBoundaryCondition)

list(APPEND HEADER_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/AttachConstraint.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/AttachConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/AffineMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/AffineMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedPlaneConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedPlaneConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedRotationConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedRotationConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/FixedTranslationConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/FixedTranslationConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/HermiteSplineConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/HermiteSplineConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/LinearVelocityConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/LinearVelocityConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/OscillatorConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/OscillatorConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ParabolicConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ParabolicConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PartialFixedConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PartialFixedConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PartialLinearMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PartialLinearMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PatchTestMovementConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PatchTestMovementConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/PositionBasedDynamicsConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/PositionBasedDynamicsConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/SkeletalMotionConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/SkeletalMotionConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToLineConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToLineConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPlaneConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPlaneConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPointConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectToPointConstraint.inl
    ${SOFABOUNDARYCONDITION_SRC}/ProjectDirectionConstraint.h
    ${SOFABOUNDARYCONDITION_SRC}/ProjectDirectionConstraint.inl
)
