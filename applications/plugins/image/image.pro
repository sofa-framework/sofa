load(sofa/pre)
defineAsPlugin(image)

TEMPLATE = lib
TARGET = sofaimage

DEFINES += SOFA_BUILD_IMAGE

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/CImg

contains(DEFINES, SOFA_EXTLIBS_FFMPEG) {
	INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/ffmpeg
	}

contains(DEFINES, SOFA_HAVE_OPENCV) {
	INCLUDEPATH += $$SOFA_OPENCV_PATH
	}

HEADERS += \
	initImage.h \
	ImageTypes.h \
	ImageContainer.h \
   	ImageViewer.h \
	ImageFilter.h \
	ImageToMeshEngine.h \
	ImageExporter.h \
	ImagePlaneWidget.h \
	ImageTransformWidget.h \
	HistogramWidget.h 

SOURCES += \
	initImage.cpp \
	ImageContainer.cpp \
	ImageViewer.cpp \
	ImageFilter.cpp \
	ImageToMeshEngine.cpp \
	ImageExporter.cpp \
	ImagePlaneWidget.cpp \
	ImageTransformWidget.cpp \
	HistogramWidget.cpp  

README_FILE = image.txt
	
load(sofa/post)
	
