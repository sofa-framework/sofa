cmake_minimum_required(VERSION 3.1)
project(SofaHaptics)

message(STATUS "SofaHaptics:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initHaptics.h
)

set(SOURCE_FILES
    initHaptics.cpp
)

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
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

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaConstraint)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTICS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
