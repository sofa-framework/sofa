include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories(${SOFA_EXTLIBS_DIR})

set(HEADER_FILES

	ImagePlaneWidget.h 
	HistogramWidget.h 
#	QImageMouseButtonsWidget.h 
	VectorVisualizationWidget.h 
	)

set(SOURCE_FILES

	initImage.cpp 
	ImageContainer.cpp 
	ImageViewer.cpp 
	ImageFilter.cpp 
    ImageTransform.cpp
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
    BranchingCellIndicesFromPositions.cpp
    BranchingCellVolumes.cpp
	)

if(SOFA_HAVE_LIBFREENECT)

    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")

endif()

set(README_FILES

	image.txt
	)

UseQt()
if(SOFA-EXTERNAL_PREFER_QT4)	
	SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
else()
	SOFA_QT3_WRAP_CPP(MOC_FILES ${HEADER_FILES})
endif()

# these header files do not need MOCcing
list(APPEND HEADER_FILES

	initImage.h 
	ImageTypes.h 
	ImageContainer.h 
   	ImageViewer.h 
	ImageFilter.h 
    ImageTransform.h
    TransferFunction.h 
    ImageValuesFromPositions.h 
    MergeImages.h 
    ImageAccumulator.h 
    DepthMapToMeshEngine.h 
    MeshToImageEngine.h 
    MarchingCubesEngine.h 
    ImageSampler.h 
    ImageExporter.h 
	ImageTransformWidget.h 
    VectorVis.h 
    ImageAlgorithms.h 
    Containers.h 
    BranchingImage.h 
    BranchingImageConverter.h
    BranchingCellIndicesFromPositions.h
    BranchingCellVolumes.h
	)

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

if(NOT SOFA-EXTERNAL_HAVE_ZLIB)
	message(AUTHOR_WARNING "Image plugin should build without zlib  but it currently doesn't. Hence we advise you to enable zlib")
endif()
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE")
AddLinkerDependencies(SofaGuiQt ${OPENGL_LIBRARIES})

if(SOFA-EXTERNAL_HAVE_OPENCV)

    AddLinkerDependencies("ml")
    AddLinkerDependencies("cvaux")
    AddLinkerDependencies("highgui")
    AddLinkerDependencies("cv")
    AddLinkerDependencies("cxcore")

endif()

if(SOFA-EXTERNAL_HAVE_LIBFREENECT)

    AddLinkerDependencies("freenect")
    AddLinkerDependencies("freenect_sync")

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
