set(SOFACONSTRAINT_SRC src/SofaConstraint)

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/BilateralConstraintResolution.h
    ${SOFACONSTRAINT_SRC}/BilateralInteractionConstraint.h
    ${SOFACONSTRAINT_SRC}/BilateralInteractionConstraint.inl
    ${SOFACONSTRAINT_SRC}/ConstraintSolverImpl.h
    ${SOFACONSTRAINT_SRC}/ConstraintStoreLambdaVisitor.h
    ${SOFACONSTRAINT_SRC}/GenericConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/GenericConstraintSolver.h
    ${SOFACONSTRAINT_SRC}/LCPConstraintSolver.h
    ${SOFACONSTRAINT_SRC}/LinearSolverConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/LinearSolverConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/PrecomputedConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/PrecomputedConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/SlidingConstraint.h
    ${SOFACONSTRAINT_SRC}/SlidingConstraint.inl
    ${SOFACONSTRAINT_SRC}/StopperConstraint.h
    ${SOFACONSTRAINT_SRC}/StopperConstraint.inl
    ${SOFACONSTRAINT_SRC}/UncoupledConstraintCorrection.h
    ${SOFACONSTRAINT_SRC}/UncoupledConstraintCorrection.inl
    ${SOFACONSTRAINT_SRC}/UniformConstraint.h
    ${SOFACONSTRAINT_SRC}/UniformConstraint.inl
    ${SOFACONSTRAINT_SRC}/UnilateralInteractionConstraint.h
    ${SOFACONSTRAINT_SRC}/UnilateralInteractionConstraint.inl
)
