include(${SOFA_CMAKE_DIR}/preProject.cmake)


# include_directories(${OPENGL_INCLUDE_DIR})

# in order to be able to include generated files (ui and moc)
include_directories(${CMAKE_CURRENT_BINARY_DIR})



UseQt()
# SET(QT_USE_QTXML TRUE)
# SET(QT_USE_QTOPENGL TRUE)



set(HEADER_FILES

	qglviewer.h 
	camera.h 
	manipulatedFrame.h 
	manipulatedCameraFrame.h 
	frame.h 
	keyFrameInterpolator.h 
	
	)
	
set(SOURCE_FILES

	qglviewer.cpp 
	camera.cpp 
	manipulatedFrame.cpp 
	manipulatedCameraFrame.cpp 
	frame.cpp 
	saveSnapshot.cpp 
	constraint.cpp 
	keyFrameInterpolator.cpp 
	mouseGrabber.cpp 
	quaternion.cpp 
	vec.cpp

	)
	
	

	
if(SOFA-EXTERNAL_PREFER_QT4)

	
	set(UI_FILES

	  "ImageInterface.ui"
# 	  VRenderInterface.Qt4.ui
	
	)	
	
	SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
	QT4_WRAP_UI(FORM_FILES ${UI_FILES})
	SOFA_QT4_WRAP_CPP(MOC2_FILES ${FORM_FILES})
else()

	# problem regarding .ui filenames
	message(FATAL_ERROR "QGLViewer project no yet converted to cmake build system with QT3!")
	
endif()



# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
ToRelativePath(MOC2_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC2_FILES}")
ToRelativePath(FORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${FORM_FILES}")


# these header files do not need MOCcing
list(APPEND HEADER_FILES

	config.h
	constraint.h 
	domUtils.h 
	mouseGrabber.h 
	quaternion.h 
	vec.h 
	)


set(MOC_FILES_GROUP "Generated")
set(FORM_FILES_GROUP "Generated")


include_directories("..")

AddCompilerDefinitions("NO_VECTORIAL_RENDER")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${MOC2_FILES} ${FORM_FILES} ${SOURCE_FILES})
AddLinkerDependencies(${OPENGL_LIBRARIES})

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

