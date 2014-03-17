list(APPEND HEADER_FILES "oclRadixSort/RadixSort.h")
list(APPEND HEADER_FILES "oclRadixSort/Scan.h")
list(APPEND HEADER_FILES "oclRadixSort/CPUSortWithOpenCL.h")
list(APPEND HEADER_FILES "oclRadixSort/CPUSortWithCuda.h")

list(APPEND SOURCE_FILES "oclRadixSort/RadixSort.cpp")
list(APPEND SOURCE_FILES "oclRadixSort/Scan.cpp")

list(APPEND OTHER_FILES "../kernels/oclRadixSort/RadixSort.cl")
list(APPEND OTHER_FILES "../kernels/oclRadixSort/Scan_b.cl")
