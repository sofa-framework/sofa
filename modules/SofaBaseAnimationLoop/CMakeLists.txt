include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
	initBaseAnimationLoop.h
    MultiStepAnimationLoop.h
    MultiTagAnimationLoop.h
	)

set(SOURCE_FILES

	initBaseAnimationLoop.cpp
    MultiStepAnimationLoop.cpp
    MultiTagAnimationLoop.cpp
	)

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_ANIMATION_LOOP")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

