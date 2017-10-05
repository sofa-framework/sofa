cmake_minimum_required(VERSION 3.1)
project(SofaGeneralAnimationLoop)

################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initGeneralAnimationLoop.h
)

set(SOURCE_FILES
    initGeneralAnimationLoop.cpp
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
        MultiStepAnimationLoop.h
        MultiTagAnimationLoop.h )

    list(APPEND SOURCE_FILES
        MultiStepAnimationLoop.cpp
        MultiTagAnimationLoop.cpp )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_ANIMATION_LOOP")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
