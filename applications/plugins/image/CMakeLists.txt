add_subdirectory(extlibs/DiffusionSolver)

cmake_minimum_required(VERSION 3.1)
project(image)

set(IMAGE_MAJOR_VERSION 0)
set(IMAGE_MINOR_VERSION 1)
set(IMAGE_VERSION ${IMAGE_MAJOR_VERSION}.${IMAGE_MINOR_VERSION})

set(SOURCE_FILES
    CollisionToCarvingEngine.cpp
    DepthMapToMeshEngine.cpp
    ImageAccumulator.cpp
    ImageContainer.cpp
    GenerateImage.cpp
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
    ImageCoordValuesFromPositions.cpp
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
    GenerateImage.h
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
    ImageCoordValuesFromPositions.h
    MarchingCubesEngine.h
    MergeImages.h
    MeshToImageEngine.h
    TransferFunction.h
    VectorVis.h
    VoronoiToMeshEngine.h
)


set(README_FILES image.txt)


find_package(SofaPython QUIET)
if( SofaPython_FOUND )

    set(PYTHON_FILES
        python/SofaImage/API.py
        python/SofaImage/__init__.py
    )

    list(APPEND SOURCE_FILES python/Binding_ImageData.cpp)
    list(APPEND HEADER_FILES python/Binding_ImageData.h python/Binding_ImageData.inl)
endif()

find_package(SofaBase REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(Newmat REQUIRED)
find_package(CImgPlugin REQUIRED)

configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/image/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/image/config.h" DESTINATION "include/image")

if( SofaPython_FOUND )
    # Config files and install rules for pythons scripts
    sofa_set_python_directory(${PROJECT_NAME} "python")
endif()

if(NOT ${SOFA_NO_OPENGL})
    list(APPEND HEADER_FILES ImageViewer.h)
    list(APPEND SOURCE_FILES ImageViewer.cpp)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Freenect QUIET)

if(FREENECT_FOUND)
    list(APPEND HEADER_FILES Kinect.h)
    list(APPEND SOURCE_FILES Kinect.cpp)
    message(STATUS "image: found the freenect library")
else()
    message(STATUS "image: could not find the freenect library, won't build the Kinect component")
endif()

# todo?

# if(SOFA-EXTERNAL_OPENCV)
# AddLinkerDependencies("ml")
# AddLinkerDependencies("cvaux")
# AddLinkerDependencies("highgui")
# AddLinkerDependencies("cv")
# AddLinkerDependencies("cxcore")
# endif()

# OpenMP is bugged in image, forcing it to be single threaded for now
if(SOFA_OPENMP)
    message( WARNING "image: deactivating OpenMP" )
    string(REPLACE "${OpenMP_CXX_FLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "${OpenMP_CXX_FLAGS}" "" CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})
endif()


find_package(DiffusionSolver CONFIG REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaComponentBase SofaGeneralVisual CImgPlugin)
target_link_libraries(${PROJECT_NAME} DiffusionSolver)
target_link_libraries(${PROJECT_NAME} newmat)

if(FREENECT_FOUND)
    target_link_libraries(${PROJECT_NAME} freenect)
    target_link_libraries(${PROJECT_NAME} freenect_sync)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CImgPlugin_INCLUDE_DIRS}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
if(FREENECT_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC "${FREENECT_INCLUDE_DIR}")
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${IMAGE_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


set(IMAGE_COMPILER_FLAGS "-DSOFA_BUILD_IMAGE")


if( SofaPython_FOUND )
    target_link_libraries(${PROJECT_NAME} SofaPython)    
    set(IMAGE_COMPILER_FLAGS "${IMAGE_COMPILER_FLAGS} -DSOFA_HAVE_SOFAPYTHON")
endif()

if(SOFA_HAVE_ZLIB)
    target_link_libraries(${PROJECT_NAME} SofaFramework::ZLIB)
endif()

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(image_test)
    endif()
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${IMAGE_COMPILER_FLAGS}")


find_package(SofaGui QUIET)
if(SofaGui_FOUND AND SOFA_GUI_QT)
    add_subdirectory(image_gui)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(image ${IMAGE_VERSION} ${PROJECT_NAME} image)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
