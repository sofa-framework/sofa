cmake_minimum_required(VERSION 3.22)
project(image VERSION 0.1)

set(PLUGIN_IMAGE_COMPILE_SET "standard" CACHE STRING "Set of image type to compile. Values are: none, standard (default), full.")
set_property(CACHE PLUGIN_IMAGE_COMPILE_SET PROPERTY STRINGS none standard full)

find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(CImgPlugin REQUIRED)

sofa_find_package(Sofa.GL QUIET)
sofa_find_package(Sofa.GUI.Qt QUIET)
sofa_find_package(SofaPython QUIET)
sofa_find_package(MultiThreading QUIET)
sofa_find_package(ZLIB)
sofa_find_package(Freenect QUIET)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(HEADER_FILES
    config.h.in
    CImgData.h
    CollisionToCarvingEngine.h
    Containers.h
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
    DepthMapToMeshEngine.h
    ImageViewer.h
)

set(SOURCE_FILES
    initImage.cpp
)

if(NOT PLUGIN_IMAGE_COMPILE_SET STREQUAL "none")
    list(APPEND SOURCE_FILES
        CollisionToCarvingEngine.cpp
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
        DepthMapToMeshEngine.cpp
        ImageViewer.cpp
    )
endif()

set(README_FILES
    image.txt
    )

if(SofaPython_FOUND)
    set(PYTHON_FILES
        python/SofaImage/API.py
        python/SofaImage/__init__.py
        )
    list(APPEND SOURCE_FILES python/Binding_ImageData.cpp)
    list(APPEND HEADER_FILES python/Binding_ImageData.h python/Binding_ImageData.inl)

    # Config files and install rules for pythons scripts
    sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python")
endif()

if(Sofa.GL_FOUND)
    if(FREENECT_FOUND)
        list(APPEND HEADER_FILES Kinect.h)
        list(APPEND SOURCE_FILES Kinect.cpp)
        message(STATUS "image: found the freenect library")
    else()
        message(STATUS "image: could not find the freenect library, won't build the Kinect component")
    endif()
endif()

# OpenMP is bugged in image, forcing it to be single threaded for now
if(SOFA_OPENMP)
    message( WARNING "image: deactivating OpenMP" )
    string(REPLACE "${OpenMP_CXX_FLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "${OpenMP_CXX_FLAGS}" "" CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})
endif()

set(PLUGIN_IMAGE_COMPILE_SET_VALUE 0) # none
if(PLUGIN_IMAGE_COMPILE_SET STREQUAL  "standard")
    set(PLUGIN_IMAGE_COMPILE_SET_VALUE 1)
elseif(PLUGIN_IMAGE_COMPILE_SET STREQUAL "full")
    set(PLUGIN_IMAGE_COMPILE_SET_VALUE 2)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_IMAGE")
target_link_libraries(${PROJECT_NAME} Sofa.Core CImgPlugin Sofa.Component.Visual)

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GL)    
endif()

if(FREENECT_FOUND)
    target_link_libraries(${PROJECT_NAME} freenect freenect_sync)
    target_include_directories(${PROJECT_NAME} PUBLIC "${FREENECT_INCLUDE_DIR}")
endif()

if(ZLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} ZLIB::ZLIB)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()

if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)    
endif()

if(MultiThreading_FOUND)
    target_link_libraries(${PROJECT_NAME} MultiThreading)
endif()

if(SOFA_BUILD_TESTS)
    add_subdirectory(image_test)
endif()

if(Sofa.GUI.Qt_FOUND)
    add_subdirectory(image_gui)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    PACKAGE_VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
)
