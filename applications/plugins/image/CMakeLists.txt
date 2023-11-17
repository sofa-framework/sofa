cmake_minimum_required(VERSION 3.12)
project(image VERSION 0.1 LANGUAGES CXX)

set(PLUGIN_IMAGE_COMPILE_SET "standard" CACHE STRING "Set of image type to compile. Values are: none, standard (default), full.")
set_property(CACHE PLUGIN_IMAGE_COMPILE_SET PROPERTY STRINGS none standard full)

find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)

sofa_find_package(Sofa.GL QUIET)
sofa_find_package(Sofa.GUI.Qt QUIET)
sofa_find_package(SofaPython QUIET)
sofa_find_package(MultiThreading QUIET)
sofa_find_package(ZLIB)
sofa_find_package(Freenect QUIET)

add_subdirectory(extlibs/CImg)

sofa_find_package(CImg REQUIRED)
# OS X only: if the user installed its own JPEG/PNG lib (typically with homebrew/port),
# it will allow to use those instead of those present in XCode's frameworks
set(CMAKE_FIND_FRAMEWORK LAST)

sofa_find_package(TIFF QUIET)
sofa_find_package(JPEG QUIET)
sofa_find_package(PNG QUIET)
sofa_find_package(ZLIB QUIET)

# Configure Paths according to detected packages
set(DEP_TARGETS "")
set(CIMG_CFLAGS -Dcimg_no_system_calls) # Remove CImg sys calls
if(TIFF_FOUND)
    message(STATUS "CImgPlugin: TIFF Image Format enabled")
    list(APPEND CIMG_CFLAGS -Dcimg_use_tiff)
    list(APPEND DEP_TARGETS TIFF::TIFF)
else()
    message(STATUS "CImgPlugin: TIFF Image Format disabled")
endif()
if(JPEG_FOUND)
    message(STATUS "CImgPlugin: JPEG Image Format enabled")
    list(APPEND CIMG_CFLAGS -Dcimg_use_jpeg)
    if(WIN32)
        # prevent INT32 redefinition on Windows (see jmorecfg.h)
        list(APPEND CIMG_CFLAGS -DXMD_H)
    endif()
    list(APPEND DEP_TARGETS JPEG::JPEG)
else()
    message(STATUS "CImgPlugin: JPEG Image Format disabled")
endif()
if(PNG_FOUND)
    message(STATUS "CImgPlugin: PNG Image Format enabled")
    list(APPEND CIMG_CFLAGS -Dcimg_use_png)
    if(UNIX)
        # Prevent pngconf error (see https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409)
        list(APPEND CIMG_CFLAGS -DPNG_SKIP_SETJMP_CHECK)
    endif()
    list(APPEND DEP_TARGETS PNG::PNG)
else()
    message(STATUS "CImgPlugin: PNG Image Format disabled")
endif()
if(ZLIB_FOUND)
    list(APPEND DEP_TARGETS ZLIB::ZLIB)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(HEADER_FILES
    src/image/config.h.in
    src/image/CImgData.h
    src/image/CollisionToCarvingEngine.h
    src/image/Containers.h
    src/image/ImageAccumulator.h
    src/image/ImageAlgorithms.h
    src/image/ImageContainer.h
    src/image/GenerateImage.h
    src/image/ImageDataDisplay.h
    src/image/ImageExporter.h
    src/image/ImageFilter.h
    src/image/ImageOperation.h
    src/image/ImageSampler.h
    src/image/ImageToRigidMassEngine.h
    src/image/ImageTransform.h
    src/image/ImageTransformEngine.h
    src/image/ImageTypes.h
    src/image/ImageValuesFromPositions.h
    src/image/ImageCoordValuesFromPositions.h
    src/image/MarchingCubesEngine.h
    src/image/MergeImages.h
    src/image/MeshToImageEngine.h
    src/image/TransferFunction.h
    src/image/VectorVis.h
    src/image/VoronoiToMeshEngine.h
    src/image/DepthMapToMeshEngine.h
    src/image/ImageViewer.h
    src/cimg/SOFACImg.h
    src/cimg/ImageCImg.h
    src/CImgPlugin/SOFACImg.h # compat
)

set(SOURCE_FILES
    src/image/initImage.cpp
)

if(NOT PLUGIN_IMAGE_COMPILE_SET STREQUAL "none")
    list(APPEND SOURCE_FILES
        src/image/CollisionToCarvingEngine.cpp
        src/image/ImageAccumulator.cpp
        src/image/ImageContainer.cpp
        src/image/GenerateImage.cpp
        src/image/ImageDataDisplay.cpp
        src/image/ImageExporter.cpp
        src/image/ImageFilter.cpp
        src/image/ImageOperation.cpp
        src/image/ImageSampler.cpp
        src/image/ImageToRigidMassEngine.cpp
        src/image/ImageTransform.cpp
        src/image/ImageTransformEngine.cpp
        src/image/ImageTypes.cpp
        src/image/ImageValuesFromPositions.cpp
        src/image/ImageCoordValuesFromPositions.cpp
        src/image/MarchingCubesEngine.cpp
        src/image/MergeImages.cpp
        src/image/MeshToImageEngine.cpp
        src/image/TransferFunction.cpp
        src/image/VoronoiToMeshEngine.cpp
        src/image/DepthMapToMeshEngine.cpp
        src/image/ImageViewer.cpp
        src/cimg/ImageCImg.cpp
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
        list(APPEND HEADER_FILES src/image/Kinect.h)
        list(APPEND SOURCE_FILES src/image/Kinect.cpp)
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
target_link_libraries(${PROJECT_NAME} Sofa.Core Sofa.Component.Visual ${DEP_TARGETS})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${CImg_INCLUDE_DIRS}>")
target_compile_options(${PROJECT_NAME} PUBLIC ${CIMG_CFLAGS})

if(SOFA_BUILD_RELEASE_PACKAGE OR CMAKE_SYSTEM_NAME STREQUAL Windows)
    sofa_install_libraries(TARGETS ${DEP_TARGETS})
endif()


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

# if(Sofa.GUI.Qt_FOUND)
#     add_subdirectory(image_gui)
# endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    PACKAGE_VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
)
