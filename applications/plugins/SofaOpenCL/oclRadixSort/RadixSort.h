/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
#include "../OpenCLProgram.h"
#include "../OpenCLKernel.h"

using namespace sofa::gpu::opencl;

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
    static OpenCLProgram *cpProgram;						// OpenCL program
    static _device_pointer d_tempKeys,d_temp_resized,d_value_resized;		// Memory objects for original keys and work space
    static _device_pointer mCounters;			// Counter for each radix
    static _device_pointer mCountersSum;		// Prefix sum of radix counters
    static _device_pointer mBlockOffsets;		// Global offsets of each radix in each block
    static OpenCLKernel *ckRadixSortBlocksKeysOnly;			// OpenCL kernels
    static OpenCLKernel *ckFindRadixOffsets;
    static OpenCLKernel *ckScanNaive;
    static OpenCLKernel *ckReorderDataKeysOnly;
    static OpenCLKernel *ckMemset;
    static _device_pointer d_tempElements;
    static unsigned int lastNumElements;

    int CTA_SIZE; // Number of threads per block
    static const unsigned int WARP_SIZE = 32;
    static const unsigned int bitStep = 4;

//    unsigned int mNumElements;     // Number of elements of temp storage allocated

    Scan *scan;

    void radixSortKeysOnly(_device_pointer d_keys,_device_pointer v, unsigned int numElements, unsigned int keyBits);
    void radixSortStepKeysOnly(_device_pointer d_keys,_device_pointer v, unsigned int nbits, unsigned int startbit, unsigned int numElements);
    void radixSortBlocksKeysOnlyOCL(_device_pointer d_keys,_device_pointer d_values,unsigned int nbits, unsigned int startbit, unsigned int numElements);
    void findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements);
    void scanNaiveOCL(unsigned int numElements);
    void reorderDataKeysOnlyOCL(_device_pointer d_keys,_device_pointer d_elements, unsigned int startbit, unsigned int numElements);

    void memset(sofa::gpu::opencl::_device_pointer dp,size_t offset,unsigned int  size);
};
#endif
