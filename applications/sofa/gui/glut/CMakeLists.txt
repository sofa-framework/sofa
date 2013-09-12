include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	SimpleGUI.h
	)

set(SOURCE_FILES

	SimpleGUI.cpp
	)

if(SOFA-EXTERNAL_BOOST)
	list(APPEND HEADER_FILES MultithreadGUI.h)
	
	list(APPEND SOURCE_FILES MultithreadGUI.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFAGUIGLUT")
AddLinkerDependencies(SofaGuiCommon ${GLUT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

