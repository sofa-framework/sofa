set(SOFABASEVISUAL_SRC src/SofaBaseVisual)
set(SOFAGENERALVISUAL_SRC src/SofaGeneralVisual)

list(APPEND HEADER_FILES
    ${SOFABASEVISUAL_SRC}/BaseCamera.h
    ${SOFABASEVISUAL_SRC}/Camera.h
    ${SOFABASEVISUAL_SRC}/InteractiveCamera.h
    ${SOFABASEVISUAL_SRC}/VisualModelImpl.h
    ${SOFABASEVISUAL_SRC}/VisualStyle.h
    ${SOFAGENERALVISUAL_SRC}/RecordedCamera.h
    ${SOFAGENERALVISUAL_SRC}/VisualTransform.h
    ${SOFAGENERALVISUAL_SRC}/Visual3DText.h
)
