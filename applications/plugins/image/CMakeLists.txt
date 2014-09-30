include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    initImage.cpp
    ImageContainer.cpp
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
    ImageDataDisplay.cpp
    VoronoiToMeshEngine.cpp
)

set(HEADER_FILES
    initImage.h
    ImageTypes.h
    ImageContainer.h
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
    ImageDataDisplay.h
    VoronoiToMeshEngine.h
)

set(README_FILES image.txt)

if (NOT SOFA-MISC_NO_OPENGL)
    list(APPEND HEADER_FILES "ImageViewer.h")
    list(APPEND SOURCE_FILES "ImageViewer.cpp")
endif()

if(SOFA-EXTERNAL_LIBFREENECT)
    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")
    AddLinkerDependencies("freenect")
    AddLinkerDependencies("freenect_sync")
endif()

if(SOFA-EXTERNAL_ZLIB)
    AddLinkerDependencies(${ZLIB_LIBRARIES})
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE")
AddLinkerDependencies(SofaBaseVisual)
# AddLinkerDependencies(${OPENGL_LIBRARIES})

if(APPLE)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build image plugin")
    endif(NOT X11_FOUND)
    AddLinkerDependencies(${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()

if(SOFA-EXTERNAL_OPENCV)
    AddLinkerDependencies("ml")
    AddLinkerDependencies("cvaux")
    AddLinkerDependencies("highgui")
    AddLinkerDependencies("cv")
    AddLinkerDependencies("cxcore")
endif()

AddLinkerDependencies(newmat)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
