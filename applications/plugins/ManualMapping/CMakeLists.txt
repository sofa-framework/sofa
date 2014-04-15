include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initManualMapping.h
    ManualLinearMapping.h
    ManualLinearMapping.inl

	)

set(SOURCE_FILES

    initManualMapping.cpp
    ManualLinearMapping.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddLinkerDependencies(SofaComponentBase)

AddCompilerDefinitions("SOFA_BUILD_ManualMapping")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
