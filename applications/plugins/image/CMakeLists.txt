set(SOFA_PROJECT_FOLDER "SofaExternal")
RegisterProjects("DiffusionSolver" PATH "${CMAKE_CURRENT_LIST_DIR}/extlibs/DiffusionSolver")

set(SOFA_PROJECT_FOLDER "SofaPlugin")
include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    CollisionToCarvingEngine.cpp
    DepthMapToMeshEngine.cpp
    ImageAccumulator.cpp
    ImageContainer.cpp
    ImageDataDisplay.cpp
    ImageExporter.cpp
    ImageFilter.cpp
    ImageOperation.cpp
    ImageSampler.cpp
    ImageToRigidMassEngine.cpp
    ImageTransform.cpp
    ImageTransformEngine.cpp
    ImageTypes.cpp
    ImageValuesFromPositions.cpp
    MarchingCubesEngine.cpp
    MergeImages.cpp
    MeshToImageEngine.cpp
    TransferFunction.cpp
    VoronoiToMeshEngine.cpp
    initImage.cpp
)

set(HEADER_FILES
    CollisionToCarvingEngine.h
    Containers.h
    DepthMapToMeshEngine.h
    ImageAccumulator.h
    ImageAlgorithms.h
    ImageContainer.h
    ImageDataDisplay.h
    ImageExporter.h
    ImageFilter.h
    ImageOperation.h
    ImageSampler.h
    ImageToRigidMassEngine.h
    ImageTransform.h
    ImageTransformEngine.h
    ImageTypes.h
    ImageValuesFromPositions.h
    MarchingCubesEngine.h
    MergeImages.h
    MeshToImageEngine.h
    TransferFunction.h
    VectorVis.h
    VoronoiToMeshEngine.h
    initImage.h
)

set(README_FILES image.txt)

# include python files
include(./python/python.cmake)

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

include_directories("${SOFA_EXTLIBS_DIR}")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})

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
AddLinkerDependencies(DiffusionSolver)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
