include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    HaptionDriver.h

	)

set(SOURCE_FILES

    initHaption.cpp 
    HaptionDriver.cpp

	)

set(README_FILES

	PluginHaption.txt
	)	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_HAPTIONPLUGIN")
set(LINKER_DEPENDENCIES SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)