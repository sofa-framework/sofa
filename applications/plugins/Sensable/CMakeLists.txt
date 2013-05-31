include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package(SofaSensableLibs REQUIRED)
include_directories(${SOFASENSABLE_INCLUDE_DIRS})

set(HEADER_FILES

    NewOmniDriver.h 
    OmniDriver.h

	)

set(SOURCE_FILES

    initSensable.cpp 
    NewOmniDriver.cpp 
    OmniDriver.cpp 

	)

set(README_FILES

	PluginSensable.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SENSABLEPLUGIN")
AddLinkerDependencies(SofaGuiQt SofaComponentGeneral SofaHaptics )

if(UNIX)
    AddLinkerDependencies("HD")
    AddLinkerDependencies("HL")
    AddLinkerDependencies("HDU")
    AddLinkerDependencies("HLU")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)