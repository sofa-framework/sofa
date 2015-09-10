include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    ManualLinearMapping.h
    ManualLinearMapping.inl
    config.h
)

set(SOURCE_FILES
    ManualLinearMapping.cpp
    initManualMapping.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddLinkerDependencies(SofaComponentBase)

AddCompilerDefinitions("SOFA_BUILD_ManualMapping")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
