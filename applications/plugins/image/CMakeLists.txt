cmake_minimum_required(VERSION 2.8)

project("SofaImage")

include(${SOFA_CMAKE_DIR}/pre.cmake)

include_directories(${SOFA_EXTLIBS_DIR})

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})


set(HEADER_FILES

	initImage.h 
	ImageTypes.h 
	ImageContainer.h 
   	ImageViewer.h 
	ImageFilter.h 
    TransferFunction.h 
    ImageValuesFromPositions.h 
    MergeImages.h 
    ImageAccumulator.h 
    DepthMapToMeshEngine.h 
    MeshToImageEngine.h 
    MarchingCubesEngine.h 
    ImageSampler.h 
    ImageExporter.h 
	ImagePlaneWidget.h 
	ImageTransformWidget.h 
	HistogramWidget.h 
#	QImageMouseButtonsWidget.h 
	VectorVisualizationWidget.h 
    VectorVis.h 
    ImageAlgorithms.h 
    Containers.h 
    BranchingImage.h 
    BranchingImageConverter.h 

	)

set(SOURCE_FILES

	initImage.cpp 
	ImageContainer.cpp 
	ImageViewer.cpp 
	ImageFilter.cpp 
    TransferFunction.cpp 
    ImageValuesFromPositions.cpp 
    MergeImages.cpp 
    ImageAccumulator.cpp 
    DepthMapToMeshEngine.cpp 
    MeshToImageEngine.cpp 
    MarchingCubesEngine.cpp 
    ImageSampler.cpp 
    ImageExporter.cpp 
	ImagePlaneWidget.cpp 
    ImageTransformWidget.cpp 
	HistogramWidget.cpp 
#	QImageMouseButtonsWidget.cpp 
    VectorVisualizationWidget.cpp 
    BranchingImageConverter.cpp 

	)

if(SOFA_HAVE_LIBFREENECT)

    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")

endif()

set(README_FILE image.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_IMAGE")
set(LINKER_DEPENDENCIES SofaGuiQt )

if(SOFA_IMAGE_HAVE_OPENCV)

    list(APPEND LINKER_DEPENDENCIES "ml")
    list(APPEND LINKER_DEPENDENCIES ""cvaux)
    list(APPEND LINKER_DEPENDENCIES "highgui")
    list(APPEND LINKER_DEPENDENCIES "cv")
    list(APPEND LINKER_DEPENDENCIES "cxcore")

endif()

if(SOFA_HAVE_LIBFREENECT)

    list(APPEND LINKER_DEPENDENCIES "freenect")
    list(APPEND LINKER_DEPENDENCIES "freenect_sync")

endif()

include(${SOFA_CMAKE_DIR}/post.cmake)