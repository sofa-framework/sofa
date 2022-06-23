set(SOFAHAPTICS_SRC src/SofaHaptics)

list(APPEND HEADER_FILES
    ${SOFAHAPTICS_SRC}/ForceFeedback.h
    ${SOFAHAPTICS_SRC}/LCPForceFeedback.h
    ${SOFAHAPTICS_SRC}/LCPForceFeedback.inl
    ${SOFAHAPTICS_SRC}/MechanicalStateForceFeedback.h
    ${SOFAHAPTICS_SRC}/NullForceFeedback.h
    ${SOFAHAPTICS_SRC}/NullForceFeedbackT.h
)
