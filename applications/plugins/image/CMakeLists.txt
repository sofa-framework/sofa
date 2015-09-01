cmake_minimum_required(VERSION 2.8.12)
project(image)

add_subdirectory(extlibs/DiffusionSolver)

set(IMAGE_MAJOR_VERSION 0)
set(IMAGE_MINOR_VERSION 0)
set(IMAGE_BUILD_VERSION 1)

set(IMAGE_VERSION ${IMAGE_MAJOR_VERSION}.${IMAGE_MINOR_VERSION}.${IMAGE_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(IMAGE_BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(IMAGE_BUILD_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(IMAGE_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(IMAGE_BUILD_CMAKE_DIR ${CMAKE_BINARY_DIR}/image)

set(IMAGE_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(IMAGE_INSTALL_CMAKE_DIR ${IMAGE_INSTALL_LIB_DIR}/cmake/image)

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

set(PYTHON_FILES

    python/SofaImage/API.py
    python/SofaImage/__init__.py

)

set(README_FILES image.txt)

find_package(SofaBase REQUIRED)
find_package(Newmat REQUIRED)
find_package(CImg REQUIRED)

set(IMAGE_DEPENDENCY_INCLUDE_DIR ${CImg_INCLUDE_DIRS} ${SofaBase_INCLUDE_DIRS} ${Newmat_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/image.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/image/image.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/image/image.h
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
endif()

# todo

# if(SOFA-EXTERNAL_OPENCV)
# AddLinkerDependencies("ml")
# AddLinkerDependencies("cvaux")
# AddLinkerDependencies("highgui")
# AddLinkerDependencies("cv")
# AddLinkerDependencies("cxcore")
# endif()

find_package(DiffusionSolver CONFIG REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaComponentBase)
target_link_libraries(${PROJECT_NAME} DiffusionSolver)
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
        EXPORT imageTargets
        RUNTIME DESTINATION "${IMAGE_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${IMAGE_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${IMAGE_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${IMAGE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(image_test)
endif()

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
