include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    init.h
    )
    
set(SOURCE_FILES

    init.cpp
    )
	
set(GROUP_BASE_DIR "..")
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT")
AddLinkerDependencies(SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
