set(SOFACONSTRAINT_SRC src/SofaConstraint)
set(SOFAUSERINTERACTION_SRC src/SofaUserInteraction)

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/AddRecordedCameraPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.h
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.inl
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.h
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/InciseAlongPathPerformer.h
    ${SOFAUSERINTERACTION_SRC}/InteractionPerformer.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.inl
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.h
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/StartNavigationPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/TopologicalChangeManager.h
)
