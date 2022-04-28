set(SOFABASEUTILS_SRC src/SofaBaseUtils)
set(SOFAGRAPHCOMPONENT_SRC src/SofaGraphComponent)

list(APPEND HEADER_FILES
    ${SOFABASEUTILS_SRC}/AddResourceRepository.h
    ${SOFABASEUTILS_SRC}/InfoComponent.h
    ${SOFABASEUTILS_SRC}/MakeAliasComponent.h
    ${SOFABASEUTILS_SRC}/MakeDataAliasComponent.h
    ${SOFABASEUTILS_SRC}/messageHandlerComponent.h
    ${SOFAGRAPHCOMPONENT_SRC}/APIVersion.h
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimation.h
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimationOnEvent.h
)
