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
#ifndef _SCAN_H_
#define _SCAN_H_

#define SCAN2

#if defined (__APPLE__) || defined(MACOSX)
//#include <OpenCL/opencl.h>
#else
//#include <CL/opencl.h>
#endif

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

using namespace sofa::helper;
using namespace sofa::gpu::opencl;

class Scan
{
public:
    Scan(
        unsigned int numElements
    );
    Scan() {}
    ~Scan();

    void scanExclusiveLarge(
        sofa::gpu::opencl::_device_pointer d_Dst,
        sofa::gpu::opencl::_device_pointer d_Src,
        unsigned int batchSize,
        unsigned int arrayLength);

private:
    static OpenCLProgram *cpProgram;                // OpenCL program
    sofa::gpu::opencl::_device_pointer d_Buffer;                     // Memory objects for original keys and work space
    static OpenCLKernel *ckScanExclusiveLocal1, *ckScanExclusiveLocal2, *ckUniformUpdate;

    static const int WORKGROUP_SIZE = 256;
    static const unsigned int   MAX_BATCH_ELEMENTS = 64 * 1048576;
    static const unsigned int MIN_SHORT_ARRAY_SIZE = 4;
    static const unsigned int MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
    static const unsigned int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
    static const unsigned int MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

    unsigned int  mNumElements;     // Number of elements of temp storage allocated

    // kernel wrappers

    void scanExclusiveLocal1(
        sofa::gpu::opencl::_device_pointer d_Dst,
        sofa::gpu::opencl::_device_pointer d_Src,
        unsigned int n,
        unsigned int size);
    void scanExclusiveLocal2(
        sofa::gpu::opencl::_device_pointer d_Buffer,
        sofa::gpu::opencl::_device_pointer d_Dst,
        sofa::gpu::opencl::_device_pointer d_Src,
        unsigned int n,
        unsigned int size);
    void uniformUpdate(
        sofa::gpu::opencl::_device_pointer d_Dst,
        sofa::gpu::opencl::_device_pointer d_Buffer,
        unsigned int n);
    static unsigned int iSnapUp(unsigned int dividend, unsigned int divisor)
    {
        return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
    }
    unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
    {
        if(!L)
        {
            log2L = 0;
            return 0;
        }
        else
        {
            for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++) {};
            return L;
        }
    }

};
#endif
