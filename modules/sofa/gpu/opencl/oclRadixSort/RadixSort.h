/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/
#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#if defined (__APPLE__) || defined(MACOSX)
//#include <OpenCL/opencl.h>
#else
//#include <CL/opencl.h>
#endif
#include "Scan.h"
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

using namespace sofa::gpu::opencl;
using namespace sofa::helper;

class RadixSort
{
public:
    RadixSort(
        unsigned int maxElements,
        const int ctaSize,
        bool keysOnly);
    RadixSort() {}
    ~RadixSort();

    void sort(sofa::gpu::opencl::_device_pointer d_keys,
            sofa::gpu::opencl::_device_pointer values,
            unsigned int  numElements,
            unsigned int  keyBits);

private:
    OpenCLProgram *cpProgram;						// OpenCL program
    _device_pointer d_tempKeys;		// Memory objects for original keys and work space
    _device_pointer mCounters;			// Counter for each radix
    _device_pointer mCountersSum;		// Prefix sum of radix counters
    _device_pointer mBlockOffsets;		// Global offsets of each radix in each block
    OpenCLKernel *ckRadixSortBlocksKeysOnly;			// OpenCL kernels
    OpenCLKernel *ckFindRadixOffsets;
    OpenCLKernel *ckScanNaive;
    OpenCLKernel *ckReorderDataKeysOnly;

    int CTA_SIZE; // Number of threads per block
    static const unsigned int WARP_SIZE = 32;
    static const unsigned int bitStep = 4;

    unsigned int  mNumElements;     // Number of elements of temp storage allocated

    Scan scan;

    void radixSortKeysOnly(_device_pointer d_keys,_device_pointer v, unsigned int numElements, unsigned int keyBits);
    void radixSortStepKeysOnly(_device_pointer d_keys,_device_pointer v, unsigned int nbits, unsigned int startbit, unsigned int numElements);
    void radixSortBlocksKeysOnlyOCL(_device_pointer d_keys,unsigned int nbits, unsigned int startbit, unsigned int numElements);
    void findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements);
    void scanNaiveOCL(unsigned int numElements);
    void reorderDataKeysOnlyOCL(_device_pointer d_keys,_device_pointer d_elements, unsigned int startbit, unsigned int numElements);
};
#endif
