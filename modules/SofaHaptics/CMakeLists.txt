include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initHaptics.h
    ForceFeedback.h
    NullForceFeedbackT.h
    NullForceFeedback.h
    #EnslavementForceFeedback.h
    LCPForceFeedback.h
    LCPForceFeedback.inl
    MechanicalStateForceFeedback.h

    )

set(SOURCE_FILES

    initHaptics.cpp
    NullForceFeedback.cpp
    NullForceFeedbackT.cpp
    #EnslavementForceFeedback.cpp
    LCPForceFeedback.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_HAPTICS" )
AddLinkerDependencies(SofaConstraint )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
