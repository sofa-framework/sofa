cmake_minimum_required(VERSION 2.8)

project("SofaHaptics")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initHaptics.h 
    controller/ForceFeedback.h 
    controller/NullForceFeedbackT.h 
    controller/NullForceFeedback.h 
    controller/EnslavementForceFeedback.h 
    controller/LCPForceFeedback.h 
    controller/LCPForceFeedback.inl 
    controller/MechanicalStateForceFeedback.h

    )
    
set(SOURCE_FILES

    initHaptics.cpp 
    controller/NullForceFeedback.cpp 
    controller/NullForceFeedbackT.cpp 
    controller/EnslavementForceFeedback.cpp 
    controller/LCPForceFeedback.cpp
 
    )
 
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_HAPTICS" )
set(LINKER_DEPENDENCIES SofaConstraint )
    
include(${SOFA_CMAKE_DIR}/post.cmake)
