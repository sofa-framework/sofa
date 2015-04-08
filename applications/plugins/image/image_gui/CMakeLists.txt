include(${SOFA_CMAKE_DIR}/preProject.cmake)

if(SOFA-LIB_GUI_QT)

# these header files need MOCcing
set(HEADER_MOC_FILES
    ImagePlaneWidget.h
    HistogramWidget.h
    # QImageMouseButtonsWidget.h
    VectorVisualizationWidget.h)

# following files do not need MOCcing
list(APPEND HEADER_FILES
    initImage_gui.h
    ImageTransformWidget.h)

set(SOURCE_FILES
    initImage_gui.cpp
    ImagePlaneWidget.cpp
    ImageTransformWidget.cpp
    HistogramWidget.cpp
    # QImageMouseButtonsWidget.cpp
    VectorVisualizationWidget.cpp)

include(../imagetoolbox/imagetoolbox.cmake)

UseQt()
SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_MOC_FILES})

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} ${SOFA_LIB_TYPE} ${HEADER_FILES} ${HEADER_MOC_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE_GUI")
AddLinkerDependencies(SofaGuiQt image ${OPENGL_LIBRARIES})

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
