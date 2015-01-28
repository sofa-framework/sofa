include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initHaptics.h 
    # ../controller/ForceFeedback.h 
    # ../controller/NullForceFeedbackT.h 
    # ../controller/NullForceFeedback.h 
    #../controller/EnslavementForceFeedback.h 
    # ../controller/LCPForceFeedback.h 
    # ../controller/LCPForceFeedback.inl 
    # ../controller/MechanicalStateForceFeedback.h

    )
    
set(SOURCE_FILES

    ../initHaptics.cpp 
    # ../controller/NullForceFeedback.cpp 
    # ../controller/NullForceFeedbackT.cpp 
    #../controller/EnslavementForceFeedback.cpp 
    # ../controller/LCPForceFeedback.cpp
 
    )
 
set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_HAPTICS" )
AddLinkerDependencies(SofaConstraint )
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
