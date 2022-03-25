set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.h
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.inl
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.h
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.inl
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.h
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.inl
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.h
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.inl
)
