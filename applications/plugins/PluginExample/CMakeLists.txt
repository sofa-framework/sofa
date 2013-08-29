include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_PLUGINEXAMPLE")
AddLinkerDependencies(SofaGuiQt)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
