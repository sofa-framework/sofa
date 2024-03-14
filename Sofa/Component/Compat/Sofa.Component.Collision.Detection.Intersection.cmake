set(SOFABASECOLLISION_SRC src/SofaBaseCollision)
set(SOFAMESHCOLLISION_SRC src/SofaMeshCollision)
set(SOFACONSTRAINT_SRC src/SofaConstraint)
set(SOFAGENERALMESHCOLLISION_SRC src/SofaGeneralMeshCollision)
set(SOFAUSERINTERACTION_SRC src/SofaUserInteraction)
set(SOFAMISCCOLLISION_SRC src/SofaMiscCollision)

list(APPEND HEADER_FILES
    ${SOFABASECOLLISION_SRC}/BaseProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/DiscreteIntersection.h
    ${SOFABASECOLLISION_SRC}/MinProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/NewProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/NewProximityIntersection.inl
    ${SOFAMESHCOLLISION_SRC}/MeshNewProximityIntersection.h
    ${SOFAMESHCOLLISION_SRC}/MeshNewProximityIntersection.inl
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.h
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.h
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.inl
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshMinProximityIntersection.h
    ${SOFAUSERINTERACTION_SRC}/RayDiscreteIntersection.h
    ${SOFAUSERINTERACTION_SRC}/RayDiscreteIntersection.inl
    ${SOFAUSERINTERACTION_SRC}/RayNewProximityIntersection.h
    ${SOFAMISCCOLLISION_SRC}/TetrahedronDiscreteIntersection.h
)