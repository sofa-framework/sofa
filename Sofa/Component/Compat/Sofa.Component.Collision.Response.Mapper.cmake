set(SOFABASECOLLISION_SRC src/SofaBaseCollision)
set(SOFAMESHCOLLISION_SRC src/SofaMeshCollision)

list(APPEND HEADER_FILES
    ${SOFABASECOLLISION_SRC}/BaseContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.inl
)