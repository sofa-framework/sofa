include(${SOFA_CMAKE_DIR}/preProject.cmake)


if(SOFA-LIB_GUI_QT)

include_directories(${SOFA_EXTLIBS_DIR})


# these header files need MOCcing
set(HEADER_FILES

	ImagePlaneWidget.h 
	HistogramWidget.h 
#	QImageMouseButtonsWidget.h 
	VectorVisualizationWidget.h 
	)

UseQt()
if(SOFA-EXTERNAL_PREFER_QT4)	
	SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
else()
	SOFA_QT3_WRAP_CPP(MOC_FILES ${HEADER_FILES})
endif()





# following files do not need MOCcing


set(SOURCE_FILES

	initImage_gui.cpp
	ImagePlaneWidget.cpp 
    ImageTransformWidget.cpp 
	HistogramWidget.cpp 
#	QImageMouseButtonsWidget.cpp 
    VectorVisualizationWidget.cpp 
	)


list(APPEND HEADER_FILES

	initImage_gui.h
	ImageTransformWidget.h 
	)

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE_GUI")
AddLinkerDependencies(SofaBaseVisual image ${OPENGL_LIBRARIES})

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
