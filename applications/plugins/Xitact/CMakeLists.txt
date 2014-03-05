include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    XiTrocarInterface.h 
    initXitact.h 
    PaceMaker.h 
    IHPDriver.h
    ITPDriver.h 
	)

set(SOURCE_FILES

    initXitact.cpp 
    PaceMaker.cpp  
    IHPDriver.cpp 
    ITPDriver.cpp 

	)

set(README_FILES

	PluginXitact.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_XITACTPLUGIN")
AddLinkerDependencies(SofaComponentMain SofaGui SofaGuiQt XiRobot)

include(${SOFA_CMAKE_DIR}/postProject.cmake)