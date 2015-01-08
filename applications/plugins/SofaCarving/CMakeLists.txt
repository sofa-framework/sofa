include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	CarvingManager.h
	initSofaCarving.h

	)

set(SOURCE_FILES

	CarvingManager.cpp
    initSofaCarving.cpp

	)

set(README_FILES

	SofaCarving.txt
	)

add_library(${PROJECT_NAME} ${SOFA_LIB_TYPE} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFACARVING")
AddLinkerDependencies(SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)