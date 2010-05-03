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

//#include <oclUtils.h>
#include "RadixSort.h"
#include "OpenCLMemoryManager.h"

extern double time1, time2, time3, time4;

RadixSort::RadixSort(
    unsigned int maxElements,
    const int ctaSize,
    bool /*keysOnly = true*/) :
    mNumElements(0),
    CTA_SIZE(ctaSize)

{

    unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ?
            (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);
    /*unsigned int numBlocks2 = ((maxElements % (CTA_SIZE * 2)) == 0) ?
            (maxElements / (CTA_SIZE * 2)) : (maxElements / (CTA_SIZE * 2) + 1);
    */
    //cl_int ciErrNum;
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&d_tempKeys, sizeof(unsigned int) * maxElements);
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mCounters,WARP_SIZE * numBlocks * sizeof(unsigned int));
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mCountersSum, WARP_SIZE * numBlocks * sizeof(unsigned int));
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mBlockOffsets,WARP_SIZE * numBlocks * sizeof(unsigned int));

#ifdef MAC
    char flags[] = {"-DMAC -cl-fast-relaxed-math"};
#else
    char flags[] = {"-cl-fast-relaxed-math"};
#endif

    cpProgram = new OpenCLProgram(OpenCLProgram::loadSource("oclRadixSort/RadixSort.cl"));
    cpProgram->buildProgram(flags);


    ckRadixSortBlocksKeysOnly = new sofa::helper::OpenCLKernel(cpProgram,"radixSortBlocksKeysOnly");
    ckFindRadixOffsets        = new sofa::helper::OpenCLKernel(cpProgram,"findRadixOffsets");
    ckScanNaive               = new sofa::helper::OpenCLKernel(cpProgram,"scanNaive");
    ckReorderDataKeysOnly     = new sofa::helper::OpenCLKernel(cpProgram,"reorderDataKeysOnly");

}

RadixSort::~RadixSort()
{
    delete(ckRadixSortBlocksKeysOnly);
    delete(ckFindRadixOffsets);
    delete(ckScanNaive);
    delete(ckReorderDataKeysOnly);
    delete(cpProgram);
    clReleaseMemObject(d_tempKeys.m);
    clReleaseMemObject(mCounters.m);
    clReleaseMemObject(mCountersSum.m);
    clReleaseMemObject(mBlockOffsets.m);
}

//------------------------------------------------------------------------
// Sorts input arrays of unsigned integer keys and (optional) values
//
// @param d_keys      Array of keys for data to be sorted
// @param values      Array of values to be sorted
// @param numElements Number of elements to be sorted.  Must be <=
//                    maxElements passed to the constructor
// @param keyBits     The number of bits in each key to use for ordering
//------------------------------------------------------------------------
void RadixSort::sort(sofa::gpu::opencl::_device_pointer d_keys,
        unsigned int *values,
        unsigned int  numElements,
        unsigned int  keyBits)
{
    if (values == 0)
    {
        radixSortKeysOnly(d_keys, numElements, keyBits);
    }
}

//----------------------------------------------------------------------------
// Main key-only radix sort function.  Sorts in place in the keys and values
// arrays, but uses the other device arrays as temporary storage.  All pointer
// parameters are device pointers.  Uses cudppScan() for the prefix sum of
// radix counters.
//----------------------------------------------------------------------------
void RadixSort::radixSortKeysOnly(sofa::gpu::opencl::_device_pointer d_keys, unsigned int numElements, unsigned int keyBits)
{
    int i = 0;
    while (keyBits > i*bitStep)
    {
        radixSortStepKeysOnly(d_keys, bitStep, i*bitStep, numElements);
        i++;
    }
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by nbits key bits per step,
// starting at startbit.
//----------------------------------------------------------------------------
void RadixSort::radixSortStepKeysOnly(sofa::gpu::opencl::_device_pointer d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    // Four step algorithms from Satish, Harris & Garland
    radixSortBlocksKeysOnlyOCL(d_keys, nbits, startbit, numElements);

    findRadixOffsetsOCL(startbit, numElements);

    scan.scanExclusiveLarge(mCountersSum, mCounters, 1, numElements/2/CTA_SIZE*16);

    reorderDataKeysOnlyOCL(d_keys, startbit, numElements);
}

//----------------------------------------------------------------------------
// Wrapper for the kernels of the four steps
//----------------------------------------------------------------------------
void RadixSort::radixSortBlocksKeysOnlyOCL(sofa::gpu::opencl::_device_pointer d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    ckRadixSortBlocksKeysOnly->setArg<cl_mem>(0, &d_keys.m);
    ckRadixSortBlocksKeysOnly->setArg<cl_mem>(1, &d_tempKeys.m);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(2, &nbits);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(3, &startbit);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(4, &numElements);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(5, &totalBlocks);
    ckRadixSortBlocksKeysOnly->setArg(6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    ckRadixSortBlocksKeysOnly->execute(0,1,NULL,globalWorkSize,localWorkSize);

}

void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    ckFindRadixOffsets->setArg<cl_mem>(0,&d_tempKeys.m);
    ckFindRadixOffsets->setArg<cl_mem>(1,&mCounters.m);
    ckFindRadixOffsets->setArg<cl_mem>(2,&mBlockOffsets.m);

    ckFindRadixOffsets->setArg<unsigned int>(3, &startbit);
    ckFindRadixOffsets->setArg<unsigned int>(4, &numElements);
    ckFindRadixOffsets->setArg<unsigned int>(5, &totalBlocks);
    ckFindRadixOffsets->setArg(6,2 * CTA_SIZE *sizeof(unsigned int), NULL);
    ckFindRadixOffsets->execute(0,1,NULL,globalWorkSize,localWorkSize);
}

#define NUM_BANKS 16
void RadixSort::scanNaiveOCL(unsigned int numElements)
{
    unsigned int nHist = numElements/2/CTA_SIZE*16;
    size_t globalWorkSize[1] = {nHist};
    size_t localWorkSize[1] = {nHist};
    unsigned int extra_space = nHist / NUM_BANKS;
    unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);

    ckScanNaive->setArg<cl_mem>(0,&mCountersSum.m);
    ckScanNaive->setArg<cl_mem>(1,&mCounters.m);
    ckScanNaive->setArg<unsigned int>(2,&nHist);
    ckScanNaive->setArg(3,2 * shared_mem_size, NULL);
    ckScanNaive->execute(0,1,NULL, globalWorkSize, localWorkSize);
}

void RadixSort::reorderDataKeysOnlyOCL(sofa::gpu::opencl::_device_pointer d_keys, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    ckReorderDataKeysOnly->setArg<cl_mem>(0,&d_keys.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(1,&d_tempKeys.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(2,&mBlockOffsets.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(3,&mCountersSum.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(4,&mCounters.m);
    ckReorderDataKeysOnly->setArg<unsigned int>(5,&startbit);
    ckReorderDataKeysOnly->setArg<unsigned int>(6,&numElements);
    ckReorderDataKeysOnly->setArg<unsigned int>(7,&totalBlocks);
    ckReorderDataKeysOnly->setArg(8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);
    ckReorderDataKeysOnly->execute(0, 1, NULL, globalWorkSize, localWorkSize);

}
