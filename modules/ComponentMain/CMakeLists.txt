include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    init.h)

set(SOURCE_FILES
    init.cpp)



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT")
AddLinkerDependencies(ComponentBase ComponentCommon ComponentGeneral ComponentAdvanced ComponentMisc)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

