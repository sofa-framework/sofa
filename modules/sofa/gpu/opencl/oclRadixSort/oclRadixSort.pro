SOURCES += oclRadixSort/RadixSort.cpp \
    oclRadixSort/Scan.cpp
HEADERS += oclRadixSort/RadixSort.h \
    oclRadixSort/Scan.h \
	oclRadixSort/CPUSortWithOpenCL.h
OTHER_FILES += ../kernels/oclRadixSort/RadixSort.cl \
    ../kernels/oclRadixSort/Scan_b.cl
