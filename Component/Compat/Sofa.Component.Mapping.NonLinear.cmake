set(SOFABASEMECHANICS_SRC src/SofaBaseMechanics)
set(SOFARIGID_SRC src/SofaRigid)
set(SOFAGENERALRIGID_SRC src/SofaGeneralRigid)
set(SOFATOPOLOGICALMAPPING_SRC src/SofaTopologyMapping)
set(SOFAMISCMAPPING_SRC src/SofaMiscMapping)

list(APPEND HEADER_FILES
    ${SOFARIGID_SRC}/RigidMapping.h
    ${SOFARIGID_SRC}/RigidMapping.inl
    ${SOFARIGID_SRC}/RigidRigidMapping.h
    ${SOFARIGID_SRC}/RigidRigidMapping.inl
    ${SOFAMISCMAPPING_SRC}/DistanceFromTargetMapping.h
    ${SOFAMISCMAPPING_SRC}/DistanceFromTargetMapping.inl
    ${SOFAMISCMAPPING_SRC}/DistanceMapping.h
    ${SOFAMISCMAPPING_SRC}/DistanceMapping.inl
    ${SOFAMISCMAPPING_SRC}/SquareDistanceMapping.h
    ${SOFAMISCMAPPING_SRC}/SquareDistanceMapping.inl
    ${SOFAMISCMAPPING_SRC}/SquareMapping.h
    ${SOFAMISCMAPPING_SRC}/SquareMapping.inl
)