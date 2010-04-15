OBJECT_DIR = OpenCLMechanicalObject


HEADERS +=OpenCLMechanicalObject.h \
	OpenCLMechanicalObject.inl

SOURCES += OpenCLMechanicalObject.cpp

OTHER_FILES += kernels/OpenCLMechanicalObject.cl
