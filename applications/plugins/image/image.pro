load(sofa/pre)
defineAsPlugin(image)

TEMPLATE = lib
TARGET = sofaimage

DEFINES += SOFA_BUILD_IMAGE SOFA_BUILD_IMAGE_GUI

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs

contains(DEFINES, SOFA_IMAGE_HAVE_OPENCV) { # should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
	INCLUDEPATH += $$SOFA_OPENCV_PATH
        LIBS += -lml  -lcvaux -lhighgui -lcv -lcxcore
        }

contains(DEFINES, SOFA_HAVE_LIBFREENECT) {
        INCLUDEPATH += $$SOFA_LIBFREENECT_PATH
        LIBS += -lfreenect -lfreenect_sync
        HEADERS += Kinect.h
        SOURCES += Kinect.cpp
        }

HEADERS += \
	initImage.h \
	ImageTypes.h \
	ImageContainer.h \
   	ImageViewer.h \
	ImageFilter.h \
        ImageOperation.h \
        ImageTransform.h \
        TransferFunction.h \
        ImageValuesFromPositions.h \
        MergeImages.h \
        ImageAccumulator.h \
        DepthMapToMeshEngine.h \
        MeshToImageEngine.h \
        MarchingCubesEngine.h \
        ImageSampler.h \
        ImageExporter.h \
	ImagePlaneWidget.h \
	image_gui/ImageTransformWidget.h \
	image_gui/HistogramWidget.h \
	image_gui/VectorVisualizationWidget.h \
        VectorVis.h \
        ImageAlgorithms.h \
        Containers.h \
        BranchingImage.h \
        BranchingImageConverter.h \
        BranchingCellIndicesFromPositions.h \
	BranchingCellOffsetsFromPositions.h \
        BranchingCellVolumes.h \
        MergeBranchingImages.h \

SOURCES += \
	initImage.cpp \
	ImageContainer.cpp \
	ImageViewer.cpp \
	ImageFilter.cpp \
        ImageOperation.cpp \
        ImageTransform.cpp \
        TransferFunction.cpp \
        ImageValuesFromPositions.cpp \
        MergeImages.cpp \
        ImageAccumulator.cpp \
        DepthMapToMeshEngine.cpp \
        MeshToImageEngine.cpp \
        MarchingCubesEngine.cpp \
        ImageSampler.cpp \
        ImageExporter.cpp \
	image_gui/ImagePlaneWidget.cpp \
	image_gui/ImageTransformWidget.cpp \
	image_gui/HistogramWidget.cpp \
    VectorVisualizationWidget.cpp \
    BranchingImageConverter.cpp \
    BranchingCellIndicesFromPositions.cpp \
    BranchingCellOffsetsFromPositions.cpp \
    BranchingCellVolumes.cpp \
    MergeBranchingImages.cpp \

README_FILE = image.txt
	
unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
	
