cmake_minimum_required(VERSION 3.1)
project(SofaHaptics)

set(HEADER_FILES
    config.h
    initHaptics.h
    )

set(SOURCE_FILES
    initHaptics.cpp
    )

list(APPEND HEADER_FILES
    # EnslavementForceFeedback.h
    ForceFeedback.h
    LCPForceFeedback.h
    LCPForceFeedback.inl
    MechanicalStateForceFeedback.h
    NullForceFeedback.h
    NullForceFeedbackT.h
    )
list(APPEND SOURCE_FILES
    # EnslavementForceFeedback.cpp
    LCPForceFeedback.cpp
    NullForceFeedback.cpp
    NullForceFeedbackT.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaConstraint)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTICS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
