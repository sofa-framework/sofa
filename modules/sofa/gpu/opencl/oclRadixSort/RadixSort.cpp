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
#define DEBUG_TEXT(t) printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);
#include "tools/showvector.h"



extern double time1, time2, time3, time4;

OpenCLKernel *RadixSort::ckRadixSortBlocksKeysOnly = NULL;			// OpenCL kernels
OpenCLKernel *RadixSort::ckFindRadixOffsets = NULL;
OpenCLKernel *RadixSort::ckScanNaive = NULL;
OpenCLKernel *RadixSort::ckReorderDataKeysOnly = NULL;
OpenCLProgram *RadixSort::cpProgram = NULL;
OpenCLKernel *RadixSort::ckMemset = NULL;

_device_pointer RadixSort::d_tempKeys;		// Memory objects for original keys and work space
_device_pointer RadixSort::mCounters;			// Counter for each radix
_device_pointer RadixSort::mCountersSum;		// Prefix sum of radix counters
_device_pointer RadixSort::mBlockOffsets;		// Global offsets of each radix in each block
_device_pointer RadixSort::d_temp_resized;
_device_pointer RadixSort::d_tempElements;

RadixSort::RadixSort(
    unsigned int maxElements,
    const int ctaSize,
    bool /*keysOnly = true*/) :
    mNumElements(0),
    CTA_SIZE(ctaSize),
    scan(maxElements/2/CTA_SIZE*16)

{
    DEBUG_TEXT("RadixSort")
    int n=1;
    while(n<maxElements)n*=2;
    maxElements=n;
    unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ?
            (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);
    /*unsigned int numBlocks2 = ((maxElements % (CTA_SIZE * 2)) == 0) ?
            (maxElements / (CTA_SIZE * 2)) : (maxElements / (CTA_SIZE * 2) + 1);
    */
    //cl_int ciErrNum;
    int sizeCounter = WARP_SIZE * numBlocks * sizeof(unsigned int);

    printf("maxelement%d\n",maxElements);
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&d_tempKeys, sizeof(unsigned int) * maxElements);
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&d_temp_resized, sizeof(unsigned int) * maxElements);
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mCounters,WARP_SIZE * numBlocks * sizeof(unsigned int));
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mCountersSum, WARP_SIZE * numBlocks * sizeof(unsigned int));
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&mBlockOffsets,WARP_SIZE * numBlocks * sizeof(unsigned int));
    sofa::gpu::opencl::OpenCLMemoryManager<unsigned int>::deviceAlloc(0,&d_tempElements, sizeof(unsigned int) * maxElements);

#ifdef MAC
    char flags[] = {"-DMAC -cl-fast-relaxed-math"};
#else
    char flags[] = {"-cl-fast-relaxed-math"};
#endif

    std::map<std::string, std::string> types;
    types["Real"]="float";
    types["Real4"]="float4";

    if(cpProgram==NULL)
    {
        cpProgram = new OpenCLProgram();
        cpProgram->setSource(*OpenCLProgram::loadSource("oclRadixSort/RadixSort.cl"));
        cpProgram->setTypes(types);
        cpProgram->createProgram();
        cpProgram->buildProgram(flags);
        std::cout << cpProgram->buildLog(0);

        ckRadixSortBlocksKeysOnly = new sofa::helper::OpenCLKernel(cpProgram,"radixSortBlocksKeysOnly");
        ckFindRadixOffsets        = new sofa::helper::OpenCLKernel(cpProgram,"findRadixOffsets");
        ckScanNaive               = new sofa::helper::OpenCLKernel(cpProgram,"scanNaive");
        ckReorderDataKeysOnly     = new sofa::helper::OpenCLKernel(cpProgram,"reorderDataKeysOnly");
        ckMemset				  = new sofa::helper::OpenCLKernel(cpProgram,"RSMemset");
    }
}

RadixSort::~RadixSort()
{
    /*	delete(ckRadixSortBlocksKeysOnly);
    	delete(ckFindRadixOffsets);
    	delete(ckScanNaive);
    	delete(ckReorderDataKeysOnly);
    	delete(cpProgram);*/
    clReleaseMemObject(d_tempKeys.m);
    clReleaseMemObject(d_temp_resized.m);
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
ShowVector *show_key;
ShowVector *show_val;

void RadixSort::sort(sofa::gpu::opencl::_device_pointer d_keys,
        sofa::gpu::opencl::_device_pointer values,
        unsigned int  numElements,
        unsigned int  keyBits)
{
    int n=1,maxElements=numElements;
    while(n<maxElements)n*=2;
    maxElements=n;

    if(show_key==NULL)show_key = new ShowVector("show_key");

    sofa::gpu::opencl::myopenclEnqueueCopyBuffer(0,d_temp_resized.m,0,d_keys.m,d_keys.offset,numElements*sizeof(uint));
    memset(d_temp_resized,numElements,maxElements-numElements);

    show_key->addOpenCLVector<int>(d_temp_resized,maxElements);
    show_key->addOpenCLVector<int>(values,numElements);

    radixSortKeysOnly(d_temp_resized,values, numElements, keyBits);

    show_key->addOpenCLVector<int>(d_temp_resized,maxElements);
    show_key->addOpenCLVector<int>(d_tempElements,maxElements);
    exit(0);


}

//----------------------------------------------------------------------------
// Main key-only radix sort function.  Sorts in place in the keys and values
// arrays, but uses the other device arrays as temporary storage.  All pointer
// parameters are device pointers.  Uses cudppScan() for the prefix sum of
// radix counters.
//----------------------------------------------------------------------------
void RadixSort::radixSortKeysOnly(sofa::gpu::opencl::_device_pointer d_keys,sofa::gpu::opencl::_device_pointer values,  unsigned int numElements, unsigned int keyBits)
{
    int i = 0;
    while (keyBits > i*bitStep)
    {
        radixSortStepKeysOnly(d_keys,values, bitStep, i*bitStep, numElements);
        i++;
    }
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by nbits key bits per step,
// starting at startbit.
//----------------------------------------------------------------------------
void RadixSort::radixSortStepKeysOnly(sofa::gpu::opencl::_device_pointer d_keys,sofa::gpu::opencl::_device_pointer d_values,  unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    int n=1;
    while(n<numElements)n*=2;
    numElements=n;

    // Four step algorithms from Satish, Harris & Garland
    radixSortBlocksKeysOnlyOCL(d_keys, d_values, nbits, startbit, numElements);

    findRadixOffsetsOCL(startbit, numElements);

    scan.scanExclusiveLarge(mCountersSum, mCounters, 1, numElements/2/CTA_SIZE*16);

    reorderDataKeysOnlyOCL(d_keys,d_values, startbit, numElements);
}

//----------------------------------------------------------------------------
// Wrapper for the kernels of the four steps
//----------------------------------------------------------------------------
ShowVector *show_sbko;


void RadixSort::radixSortBlocksKeysOnlyOCL(sofa::gpu::opencl::_device_pointer d_keys,sofa::gpu::opencl::_device_pointer d_values, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    DEBUG_TEXT("radixSortBlocksKeysOnlyOCL")
    ERROR_OFFSET(d_keys);
    ERROR_OFFSET(d_tempKeys);

    if(show_sbko==NULL)show_sbko = new ShowVector("show_sbko");
    show_sbko->addTitle("d_keys before");
    show_sbko->addOpenCLVector<int>(d_keys,8000);
    show_sbko->addTitle("d_tempKeys before");
    show_sbko->addOpenCLVector<int>(d_tempKeys,numElements);
    /*show_sbko->addTitle("d_values before");
    show_sbko->addOpenCLVector<int>(d_values,8000);
    show_sbko->addTitle("d_tempvalues before");
    show_sbko->addOpenCLVector<int>(d_tempElements,numElements);*/

    BARRIER(d_keys,__FILE__,__LINE__);
    unsigned int totalBlocks = numElements/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    ckRadixSortBlocksKeysOnly->setArg<cl_mem>(0, &d_keys.m);
    ckRadixSortBlocksKeysOnly->setArg<cl_mem>(1, &d_tempKeys.m);
//	ckRadixSortBlocksKeysOnly->setArg<cl_mem>(2, &d_values.m);
//	ckRadixSortBlocksKeysOnly->setArg<cl_mem>(3, &d_tempElements.m);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(2, &nbits);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(3, &startbit);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(4, &numElements);
    ckRadixSortBlocksKeysOnly->setArg<unsigned int>(5, &totalBlocks);
    ckRadixSortBlocksKeysOnly->setArg(6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
//	ckRadixSortBlocksKeysOnly->setArg(9, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    ckRadixSortBlocksKeysOnly->execute(0,1,NULL,globalWorkSize,localWorkSize);

    show_sbko->addTitle("d_keys after");
    show_sbko->addOpenCLVector<int>(d_keys,8000);
    show_sbko->addTitle("d_tempKeys after");
    show_sbko->addOpenCLVector<int>(d_tempKeys,numElements);
    /*show_sbko->addTitle("d_values after");
    show_sbko->addOpenCLVector<int>(d_values,8000);
    show_sbko->addTitle("d_tempvalues after");
    show_sbko->addOpenCLVector<int>(d_tempElements,numElements);*/

    BARRIER(d_keys,__FILE__,__LINE__);
    DEBUG_TEXT("~radixSortBlocksKeysOnlyOCL")
}

void RadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
{
    DEBUG_TEXT("findRadixOffsetsOCL")
    ERROR_OFFSET(d_tempKeys);
    ERROR_OFFSET(mCounters);
    ERROR_OFFSET(mBlockOffsets);

    BARRIER(d_tempKeys,__FILE__,__LINE__);
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
    BARRIER(d_tempKeys,__FILE__,__LINE__);
    DEBUG_TEXT("~findRadixOffsetsOCL")
}

#define NUM_BANKS 16
void RadixSort::scanNaiveOCL(unsigned int numElements)
{
    DEBUG_TEXT("scanNaiveOCL")
    ERROR_OFFSET(mCountersSum);
    ERROR_OFFSET(mCounters);

    BARRIER(mCountersSum,__FILE__,__LINE__);

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

    BARRIER(mCountersSum,__FILE__,__LINE__);
    DEBUG_TEXT("~scanNaiveOCL")
}

void RadixSort::reorderDataKeysOnlyOCL(sofa::gpu::opencl::_device_pointer d_keys,sofa::gpu::opencl::_device_pointer d_elements, unsigned int startbit, unsigned int numElements)
{
    DEBUG_TEXT("reorderDataKeysOnlyOCL")
    ERROR_OFFSET(d_keys);
    ERROR_OFFSET(d_tempKeys);
//	ERROR_OFFSET(d_elements);
    ERROR_OFFSET(mBlockOffsets);
    ERROR_OFFSET(mCountersSum);
    ERROR_OFFSET(mCounters);
    BARRIER(d_keys,__FILE__,__LINE__);
    unsigned int totalBlocks = numElements/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    ckReorderDataKeysOnly->setArg<cl_mem>(0,&d_keys.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(1,&d_tempKeys.m);
//	ckReorderDataKeysOnly->setArg<cl_mem>(2,&d_elements.m);
//	ckReorderDataKeysOnly->setArg<unsigned int>(3,(unsigned int*)&d_elements.offset);
//	ckReorderDataKeysOnly->setArg<cl_mem>(4,&d_tempElements.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(2,&mBlockOffsets.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(3,&mCountersSum.m);
    ckReorderDataKeysOnly->setArg<cl_mem>(4,&mCounters.m);
    ckReorderDataKeysOnly->setArg<unsigned int>(5,&startbit);
    ckReorderDataKeysOnly->setArg<unsigned int>(6,&numElements);
    ckReorderDataKeysOnly->setArg<unsigned int>(7,&totalBlocks);
    ckReorderDataKeysOnly->setArg(8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);

    ckReorderDataKeysOnly->execute(0, 1, NULL, globalWorkSize, localWorkSize);
    BARRIER(d_keys,__FILE__,__LINE__);
    DEBUG_TEXT("~reorderDataKeysOnlyOCL")
}

void RadixSort::memset(sofa::gpu::opencl::_device_pointer dp,size_t offset,uint size)
{
    int BSIZE=32;

    ckMemset->setArg<cl_mem>(0,&dp.m);
    ckMemset->setArg<unsigned int>(1,(unsigned int*)&offset);

    size_t globalWorkSize[1] = {((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1)};
    size_t localWorkSize[1] = {BSIZE};



    ckMemset->execute(0,1,NULL,globalWorkSize,NULL);
}
