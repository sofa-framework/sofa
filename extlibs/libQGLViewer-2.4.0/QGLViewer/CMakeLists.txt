include(${SOFA_CMAKE_DIR}/preProject.cmake)


# in order to be able to include generated files (ui and moc)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

UseQt()

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
	
	VRender/BackFaceCullingOptimizer.cpp
	VRender/EPSExporter.cpp
	VRender/FIGExporter.cpp
	VRender/NVector3.cpp
	VRender/Primitive.cpp
	VRender/TopologicalSortMethod.cpp
	VRender/Vector3.cpp
	VRender/VRender.cpp
	VRender/BSPSortMethod.cpp
	VRender/Exporter.cpp
	VRender/gpc.cpp
	VRender/ParserGL.cpp
	VRender/PrimitivePositioning.cpp
	VRender/Vector2.cpp
	VRender/VisibilityOptimizer.cpp
	)

set(UI_FILES

	ImageInterface.ui
 	VRenderInterface.ui
	
	)	
	
#SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
#QT4_WRAP_UI(FORM_FILES ${UI_FILES})
SOFA_QT_WRAP_CPP(MOC_FILES ${HEADER_FILES})
SOFA_QT_WRAP_UI(FORM_FILES ${UI_FILES})


# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
#ToRelativePath(MOC_UI_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_UI_FILES}")
ToRelativePath(FORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${FORM_FILES}")

# these header files do not need MOCcing
list(APPEND HEADER_FILES

	config.h
	constraint.h 
	domUtils.h 
	mouseGrabber.h 
	quaternion.h 
	vec.h 
	
	VRender/AxisAlignedBox.h
	VRender/Exporter.h
	VRender/gpc.h
	VRender/NVector3.h
	VRender/Optimizer.h
	VRender/ParserGL.h
	VRender/Primitive.h
	VRender/PrimitivePositioning.h
	VRender/SortMethod.h
	VRender/Types.h
	VRender/Vector2.h
	VRender/Vector3.h
	VRender/VRender.h
	)

set(MOC_FILES_GROUP "Generated")
set(MOC_UI_FILES_GROUP "Generated")
set(FORM_FILES_GROUP "Generated")

include_directories("..")

if(WIN32)
	AddCompilerDefinitions(CREATE_QGLVIEWER_DLL)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${MOC_UI_FILES} ${FORM_FILES} ${SOURCE_FILES})
AddLinkerDependencies(${OPENGL_LIBRARIES})

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

