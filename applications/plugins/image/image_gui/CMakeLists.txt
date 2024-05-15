cmake_minimum_required(VERSION 3.22)
project(image_gui)

set(HEADER_FILES
    src/image_gui/ImageTransformWidget.h
    src/image_gui/config.h.in
)

set(MOC_HEADER_FILES
    src/image_gui/HistogramWidget.h
    src/image_gui/ImagePlaneWidget.h
    src/image_gui/VectorVisualizationWidget.h
)

set(SOURCE_FILES
    src/image_gui/HistogramWidget.cpp
    src/image_gui/ImagePlaneWidget.cpp
    src/image_gui/ImageTransformWidget.cpp
    src/image_gui/VectorVisualizationWidget.cpp
    src/image_gui/initImage_gui.cpp
)

include(../imagetoolbox/imagetoolbox.cmake)

find_package(Sofa.GUI.Qt REQUIRED QUIET)
find_package(Qt5 COMPONENTS Core QUIET)
if (NOT Qt5Core_FOUND)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
        find_package(Qt6 COMPONENTS Core CoreTools REQUIRED QUIET)
        if (Qt6Core_FOUND)
            qt6_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
        endif()
        # else() should never happened as it would already break in Sofa.GUI.Qt
    endif()
else()
    qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
endif()

set(MOC_FILES_GROUP "Generated")

add_definitions("-DSOFA_BUILD_IMAGE_GUI")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} image ${OPENGL_LIBRARIES} Sofa.GUI.Qt)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
