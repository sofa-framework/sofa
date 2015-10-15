cmake_minimum_required(VERSION 2.8.12)
project(image_gui)

set(HEADER_FILES
    ImageTransformWidget.h
    config.h
)

set(MOC_HEADER_FILES
    HistogramWidget.h
    ImagePlaneWidget.h
    VectorVisualizationWidget.h
)

set(SOURCE_FILES
    HistogramWidget.cpp
    ImagePlaneWidget.cpp
    ImageTransformWidget.cpp
    VectorVisualizationWidget.cpp
    initImage_gui.cpp
)

include(../imagetoolbox/imagetoolbox.cmake)

find_package(SofaGui REQUIRED QUIET)

qt4_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
set(MOC_FILES_GROUP "Generated")

add_definitions("-DSOFA_BUILD_IMAGE_GUI")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_link_libraries(image_gui image ${OPENGL_LIBRARIES} SofaGuiQt)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

# TODO: install rules
