set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)
set(SOFAMISCENGINE_SRC src/SofaMiscEngine)

list(APPEND HEADER_FILES
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.h
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.inl
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.h
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.inl
    ${SOFAGENERALENGINE_SRC}/TransformEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformEngine.inl
    ${SOFAGENERALENGINE_SRC}/TransformMatrixEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.inl
    ${SOFAGENERALENGINE_SRC}/DifferenceEngine.h
    ${SOFAGENERALENGINE_SRC}/DifferenceEngine.inl
    ${SOFAGENERALENGINE_SRC}/MathOp.h
    ${SOFAGENERALENGINE_SRC}/MathOp.inl
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.h
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.inl
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.h
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.inl
    ${SOFAGENERALENGINE_SRC}/MapIndices.h
    ${SOFAGENERALENGINE_SRC}/MapIndices.inl    
    ${SOFAGENERALENGINE_SRC}/ROIValueMapper.h
    ${SOFAGENERALENGINE_SRC}/DilateEngine.h
    ${SOFAGENERALENGINE_SRC}/DilateEngine.inl
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.h
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.inl    
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.h
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.inl
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.h
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.inl
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.h
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.inl
)
