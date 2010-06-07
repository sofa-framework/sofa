SOURCES += oclRadixSort/RadixSort.cpp \
    oclRadixSort/Scan.cpp
HEADERS += oclRadixSort/RadixSort.h \
    oclRadixSort/Scan.h \
    oclRadixSort/CPUSortWithOpenCL.h \
	oclRadixSort/CPUSortWithCuda.h
OTHER_FILES += ../kernels/oclRadixSort/RadixSort.cl \
    ../kernels/oclRadixSort/Scan_b.cl
