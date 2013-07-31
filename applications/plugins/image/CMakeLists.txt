include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories(${SOFA_EXTLIBS_DIR})



set(SOURCE_FILES

	initImage.cpp 
	ImageContainer.cpp 
	ImageViewer.cpp 
	ImageFilter.cpp 
	ImageOperation.cpp 
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
    BranchingImageConverter.cpp 
    BranchingCellIndicesFromPositions.cpp
    BranchingCellOffsetsFromPositions.cpp
    BranchingCellVolumes.cpp
    MergeBranchingImages.cpp
	)



list(APPEND HEADER_FILES

	initImage.h 
	ImageTypes.h 
	ImageContainer.h 
   	ImageViewer.h 
	ImageFilter.h 
	ImageOperation.h 
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
    VectorVis.h 
    ImageAlgorithms.h 
    Containers.h 
    BranchingImage.h 
    BranchingImageConverter.h
    BranchingCellIndicesFromPositions.h
    BranchingCellOffsetsFromPositions.h
    BranchingCellVolumes.h
    MergeBranchingImages.h
	)

	
if(SOFA_HAVE_LIBFREENECT)

    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")
    
    
    AddLinkerDependencies("freenect")
    AddLinkerDependencies("freenect_sync")

endif()

set(README_FILES

	image.txt
      )


if(NOT SOFA-EXTERNAL_HAVE_ZLIB)
	message(AUTHOR_WARNING "Image plugin should build without zlib  but it currently doesn't. Hence we advise you to enable zlib")
endif()
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE")
# AddLinkerDependencies(SofaBaseVisual ${OPENGL_LIBRARIES})
AddLinkerDependencies(SofaBaseVisual)

if(SOFA-EXTERNAL_HAVE_OPENCV)

    AddLinkerDependencies("ml")
    AddLinkerDependencies("cvaux")
    AddLinkerDependencies("highgui")
    AddLinkerDependencies("cv")
    AddLinkerDependencies("cxcore")

endif()

AddLinkerDependencies(newmat)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
