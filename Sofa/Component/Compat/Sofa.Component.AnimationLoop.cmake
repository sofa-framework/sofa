set(SOFACONSTRAINT_SRC src/SofaConstraint)
set(SOFAGENERALANIMATIONLOOP_SRC src/SofaGeneralAnimationLoop)

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/FreeMotionAnimationLoop.h
    ${SOFACONSTRAINT_SRC}/FreeMotionTask.h
    ${SOFACONSTRAINT_SRC}/ConstraintAnimationLoop.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiStepAnimationLoop.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiTagAnimationLoop.h
)
