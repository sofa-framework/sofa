set(SOFABASEVISUAL_SRC src/SofaBaseVisual)
set(SOFAGRAPHCOMPONENT_SRC src/SofaGraphComponent)

list(APPEND HEADER_FILES
    ${SOFABASEVISUAL_SRC}/BackgroundSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/AddFrameButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/AddRecordedCameraButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/AttachBodyButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/FixPickedParticleButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/MouseButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/SofaDefaultPathSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/StatsSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/ViewerSetting.h
)
