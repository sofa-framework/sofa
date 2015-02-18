cmake_minimum_required(VERSION 2.8.12)
project(image)

set(IMAGE_MAJOR_VERSION 0)
set(IMAGE_MINOR_VERSION 0)
set(IMAGE_BUILD_VERSION 1)

set(IMAGE_VERSION ${IMAGE_MAJOR_VERSION}.${IMAGE_MINOR_VERSION}.${IMAGE_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(IMAGE_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(IMAGE_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(IMAGE_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(IMAGE_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/image)

set(IMAGE_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_CMAKE_DIR   ${IMAGE_INSTALL_LIB_DIR}/cmake/image )

set(SOURCE_FILES
    initImage.cpp
    ImageContainer.cpp
    ImageFilter.cpp
    ImageOperation.cpp
    ImageTransform.cpp
    ImageTransformEngine.cpp
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
    ImageToRigidMassEngine.cpp)

set(HEADER_FILES
    initImage.h
    ImageTypes.h
    ImageContainer.h
    ImageFilter.h
    ImageOperation.h
    ImageTransform.h
    ImageTransformEngine.h
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
    ImageToRigidMassEngine.h)

set(README_FILES image.txt)

# include python files
include(./python/python.cmake)

find_package(SofaBase REQUIRED)
find_package(Newmat REQUIRED)

set(IMAGE_DEPENDENCY_INCLUDE_DIR ${SofaBase_INCLUDE_DIRS} ${Newmat_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/image.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/image/image.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/image/image.h
         DESTINATION "${IMAGE_INSTALL_INCLUDE_DIR}/image")

if (NOT SOFA-NO_OPENGL)
    list(APPEND HEADER_FILES "ImageViewer.h")
    list(APPEND SOURCE_FILES "ImageViewer.cpp")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Freenect)

if(FREENECT_FOUND)
    list(APPEND IMAGE_DEPENDENCY_INCLUDE_DIR ${FREENECT_INCLUDE_DIR})
    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")
    AddLinkerDependencies("freenect")
    AddLinkerDependencies("freenect_sync")
endif()

# todo

# if(SOFA-EXTERNAL_OPENCV)
#     AddLinkerDependencies("ml")
#     AddLinkerDependencies("cvaux")
#     AddLinkerDependencies("highgui")
#     AddLinkerDependencies("cv")
#     AddLinkerDependencies("cxcore")
# endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaComponentBase)
target_link_libraries(${PROJECT_NAME} newmat)

if(FREENECT_FOUND)
    target_link_libraries(${PROJECT_NAME} freenect)
    target_link_libraries(${PROJECT_NAME} freenect_sync)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${IMAGE_DEPENDENCY_INCLUDE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${IMAGE_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_IMAGE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})

if(SOFA_HAVE_ZLIB)
    target_link_libraries(${PROJECT_NAME} SofaFramework::ZLIB)
endif()

if(APPLE)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build plugin 'image'")
    endif()
    target_link_libraries(${PROJECT_NAME} ${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()


install(TARGETS ${PROJECT_NAME}
        COMPONENT image_libraries
        EXPORT  imageTargets
        RUNTIME DESTINATION "${IMAGE_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${IMAGE_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${IMAGE_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${IMAGE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")


include(CMakePackageConfigHelpers)
# imageConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${IMAGE_BUILD_CMAKE_DIR}/imageConfigVersion.cmake
                                 VERSION ${IMAGE_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${IMAGE_BUILD_CMAKE_DIR}/imageConfigVersion.cmake
        DESTINATION ${IMAGE_INSTALL_CMAKE_DIR})


# imageConfig.cmake (build tree)
set(IMAGE_INCLUDE_DIR "${IMAGE_BUILD_INCLUDE_DIR}")
configure_package_config_file(imageConfig.cmake.in
                              ${IMAGE_BUILD_CMAKE_DIR}/imageConfig.cmake
                              INSTALL_DESTINATION ${IMAGE_BUILD_CMAKE_DIR}
                              PATH_VARS IMAGE_INCLUDE_DIR)

# imageConfig.cmake (installed)
set(IMAGE_INCLUDE_DIR "${IMAGE_INSTALL_INCLUDE_DIR}")
configure_package_config_file(imageConfig.cmake.in
                              ${IMAGE_BUILD_CMAKE_DIR}/imageConfigInstall.cmake
                              INSTALL_DESTINATION ${IMAGE_INSTALL_CMAKE_DIR}
                              PATH_VARS IMAGE_INCLUDE_DIR)

install(FILES ${IMAGE_BUILD_CMAKE_DIR}/imageConfigInstall.cmake
        DESTINATION ${IMAGE_INSTALL_CMAKE_DIR}
        RENAME imageConfig.cmake)

# imageTargets.cmake (installed)
install(EXPORT imageTargets
        DESTINATION ${IMAGE_INSTALL_CMAKE_DIR})
