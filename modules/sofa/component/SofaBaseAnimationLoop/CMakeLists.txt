include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	../initBaseAnimationLoop.h
    # ../animationloop/MultiStepAnimationLoop.h
    # ../animationloop/MultiTagAnimationLoop.h
	)

set(SOURCE_FILES

	../initBaseAnimationLoop.cpp
    # ../animationloop/MultiStepAnimationLoop.cpp
    # ../animationloop/MultiTagAnimationLoop.cpp
	)

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_ANIMATION_LOOP")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

