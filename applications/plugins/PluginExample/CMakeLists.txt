include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    MyBehaviorModel.h 
    MyDataWidgetUnsigned.h 
    MyMappingPendulumInPlane.h 
    MyMappingPendulumInPlane.inl 
    MyProjectiveConstraintSet.h 
    MyProjectiveConstraintSet.inl 
	initPlugin.h

	)

set(SOURCE_FILES

    MyBehaviorModel.cpp 
    MyDataWidgetUnsigned.cpp 
    MyProjectiveConstraintSet.cpp 
    MyMappingPendulumInPlane.cpp 
    initPlugin.cpp

	)

set(README_FILES

	PluginExample.txt
	)
	
UseQt()	
if(SOFA-EXTERNAL_PREFER_QT4)
	SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
else()
	SOFA_QT3_WRAP_CPP(MOC_FILES ${HEADER_FILES})
endif()

# make mocced header files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_PLUGINEXAMPLE")
AddLinkerDependencies(SofaGuiQt)

include(${SOFA_CMAKE_DIR}/postProject.cmake)