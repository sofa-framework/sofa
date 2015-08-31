include(${SOFA_CMAKE_DIR}/preProject.cmake)

if(SOFA-LIB_GUI_QT)

# these header files need MOCcing
set(HEADER_MOC_FILES
    HistogramWidget.h
    ImagePlaneWidget.h
    # QImageMouseButtonsWidget.h
    VectorVisualizationWidget.h
)

# following files do not need MOCcing
list(APPEND HEADER_FILES
    ImageTransformWidget.h
    initImage_gui.h
)

set(SOURCE_FILES
    HistogramWidget.cpp
    ImagePlaneWidget.cpp
    ImageTransformWidget.cpp
    # QImageMouseButtonsWidget.cpp
    VectorVisualizationWidget.cpp
    initImage_gui.cpp
)

include(../imagetoolbox/imagetoolbox.cmake)

UseQt()
SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_MOC_FILES})

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${HEADER_MOC_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE_GUI")
AddLinkerDependencies(SofaGuiMain image ${OPENGL_LIBRARIES})

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
