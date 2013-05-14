include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	qglviewer.h 
	camera.h 
	manipulatedFrame.h 
	manipulatedCameraFrame.h 
	frame.h 
	constraint.h 
	keyFrameInterpolator.h 
	mouseGrabber.h 
	quaternion.h 
	vec.h 
	domUtils.h 
	config.h
	
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

# AHEM.... todo later :-)

message(FATAL_ERROR "QGLViewer project no yet converted to cmake build system !")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    set(COMPILER_FLAGS "-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

