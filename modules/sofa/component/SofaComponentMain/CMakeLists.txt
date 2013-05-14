include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../init.h
    )
    
set(SOURCE_FILES

    ../init.cpp
    )
	
set(GROUP_BASE_DIR "..")
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT")
set(LINKER_DEPENDENCIES SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
