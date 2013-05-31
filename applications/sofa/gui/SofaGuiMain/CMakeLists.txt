include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	../Main.h
	)

set(SOURCE_FILES

	../Main.cpp
	)
	
set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_GUIMAIN")
AddLinkerDependencies(SofaGuiCommon)

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
    AddLinkerDependencies(SofaGuiQt)
endif()

if(SOFA-LIB_GUI_GLUT)
    AddLinkerDependencies(SofaGuiGlut)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

