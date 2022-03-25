set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)
set(SOFAMISCENGINE_SRC src/SofaMiscEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/TransformEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformEngine.inl
    ${SOFAGENERALENGINE_SRC}/TransformMatrixEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.inl
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.h
    ${SOFAMISCENGINE_SRC}/DisplacementMatrixEngine.inl
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.h
    ${SOFAMISCENGINE_SRC}/ProjectiveTransformEngine.inl
)
