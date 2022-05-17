set(SOFABASECOLLISION_SRC src/SofaBaseCollision)
set(SOFAMESHCOLLISION_SRC src/SofaMeshCollision)
set(SOFACONSTRAINT_SRC src/SofaConstraint)
set(SOFAUSERINTERACTION_SRC src/SofaUserInteraction)
set(SOFAOBJECTINTERACTION_SRC src/SofaObjectInteraction)
set(SOFAMISCCOLLISION_SRC src/SofaMiscCollision)

list(APPEND HEADER_FILES
    ${SOFABASECOLLISION_SRC}/ContactListener.h
    ${SOFABASECOLLISION_SRC}/DefaultContactManager.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.inl
    ${SOFACONSTRAINT_SRC}/ContactIdentifier.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.inl
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.h
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.inl
    ${SOFAUSERINTERACTION_SRC}/RayContact.h
    ${SOFAOBJECTINTERACTION_SRC}/PenalityContactForceField.h
    ${SOFAOBJECTINTERACTION_SRC}/PenalityContactForceField.inl
    ${SOFAMISCCOLLISION_SRC}/BarycentricStickContact.h
    ${SOFAMISCCOLLISION_SRC}/BarycentricStickContact.inl
    ${SOFAMISCCOLLISION_SRC}/RuleBasedContactManager.h
)