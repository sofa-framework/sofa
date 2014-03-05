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


include_directories("${SOFA_EXTLIBS_DIR}")
# include_directories("${SOFA_APPLICATIONS_PLUGINS_DIR}")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddLinkerDependencies(SofaComponentBase)

AddCompilerDefinitions("SOFA_BUILD_ManualMapping")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
