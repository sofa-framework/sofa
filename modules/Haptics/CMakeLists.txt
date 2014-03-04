include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initHaptics.h
    ForceFeedback.h
    NullForceFeedbackT.h
    NullForceFeedback.h
    LCPForceFeedback.h
    LCPForceFeedback.inl
    MechanicalStateForceFeedback.h)

set(SOURCE_FILES
    initHaptics.cpp
    NullForceFeedback.cpp
    NullForceFeedbackT.cpp
    LCPForceFeedback.cpp)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_HAPTICS")
AddLinkerDependencies(Constraint)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

