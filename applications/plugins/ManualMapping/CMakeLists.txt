include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    ManualMapping.h
    ManualLinearMapping.h
    ManualLinearMapping.inl)

set(SOURCE_FILES
    ManualMapping.cpp)

sofa_add_plugin(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddLinkerDependencies(SofaComponentBase)

AddCompilerDefinitions("SOFA_BUILD_ManualMapping")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
