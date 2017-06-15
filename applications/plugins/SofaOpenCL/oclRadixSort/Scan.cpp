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


#include "../OpenCLMemoryManager.h"
#include "Scan.h"

OpenCLProgram *Scan::cpProgram=NULL;                // OpenCL program
OpenCLKernel *Scan::ckScanExclusiveLocal1=NULL;
OpenCLKernel *Scan::ckScanExclusiveLocal2=NULL;
OpenCLKernel *Scan::ckUniformUpdate=NULL;


#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

Scan::Scan(
    unsigned int numElements
) :
    mNumElements(numElements)
{

//	cl_int ciErrNum;
    if (numElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)
    {
        OpenCLMemoryManager<cl_uint>::deviceAlloc(0,&d_Buffer, numElements / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * sizeof(cl_uint));
    }

    //shrLog("Create and build Scan program\n");
//	size_t szKernelLength; // Byte size of kernel code

    if(cpProgram==NULL)
    {
        cpProgram = new OpenCLProgram("oclRadixSort/Scan_b.cl");
        std::cout << "\n-/--/--/--/-\n" << cpProgram->buildLog(0) << "\n/-//-//-//-/\n";
        cpProgram->buildProgram();

        ckScanExclusiveLocal1 = new OpenCLKernel(cpProgram,"scanExclusiveLocal1");
        ckScanExclusiveLocal2 = new OpenCLKernel(cpProgram,"scanExclusiveLocal2");
        ckUniformUpdate =  new OpenCLKernel(cpProgram,"uniformUpdate");

    }
}

Scan::~Scan()
{
//	cl_int ciErrNum;
    /*	delete(ckScanExclusiveLocal1);
    	delete(ckScanExclusiveLocal2);
    	delete(ckUniformUpdate);
    */
    if (mNumElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)
    {
        clReleaseMemObject(d_Buffer.m);
    }
    delete(cpProgram);

}

// main exclusive scan routine
void Scan::scanExclusiveLarge(
    sofa::gpu::opencl::_device_pointer d_Dst,
    sofa::gpu::opencl::_device_pointer d_Src,
    unsigned int batchSize,
    unsigned int arrayLength
)
{

    ERROR_OFFSET(d_Dst);
    ERROR_OFFSET(d_Src);
    //Check power-of-two factorization
    unsigned int log2L;
    unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);

    if(!(factorizationRemainder == 1)) {printf("Error: %s %d\n",__FILE__,__LINE__); exit(-1);}

    //Check supported size range
    if(!((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE))) {printf("Error: %s %d -> (%d>=%d)&&(%d<=%d)\n",__FILE__,__LINE__,arrayLength,MIN_LARGE_ARRAY_SIZE,arrayLength,MAX_LARGE_ARRAY_SIZE); exit(-1);}

    //Check total batch size limit
    if(!((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS))printf("Error: %s %d -> (%d*%d)<=%d\n",__FILE__,__LINE__,batchSize,arrayLength,MAX_BATCH_ELEMENTS);

    scanExclusiveLocal1(
        d_Dst,
        d_Src,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
        4 * WORKGROUP_SIZE
    );

    scanExclusiveLocal2(
        d_Buffer,
        d_Dst,
        d_Src,
        batchSize,
        arrayLength / (4 * WORKGROUP_SIZE)
    );

    uniformUpdate(
        d_Dst,
        d_Buffer,
        (batchSize * arrayLength) / (4 * WORKGROUP_SIZE)
    );

}


void Scan::scanExclusiveLocal1(
    sofa::gpu::opencl::_device_pointer d_Dst,
    sofa::gpu::opencl::_device_pointer d_Src,
    unsigned int n,
    unsigned int size
)
{
    DEBUG_TEXT("scanExclusiveLocal1");
    BARRIER(d_Dst,__FILE__,__LINE__);

//   cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    ckScanExclusiveLocal1->setArg<cl_mem>(0,&d_Dst.m);

    ckScanExclusiveLocal1->setArg<_device_pointer>( 1, &d_Src);

    ckScanExclusiveLocal1->setArg(2, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);

    ckScanExclusiveLocal1->setArg<unsigned int>( 3 , &size);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = (n * size) / 4;

    ckScanExclusiveLocal1->execute(0, 1, NULL, &globalWorkSize, &localWorkSize);

//   oclCheckError(ciErrNum, CL_SUCCESS);

    BARRIER(d_Dst,__FILE__,__LINE__);
    DEBUG_TEXT("~scanExclusiveLocal1");
}

void Scan::scanExclusiveLocal2(
    sofa::gpu::opencl::_device_pointer d_Buffer,
    sofa::gpu::opencl::_device_pointer d_Dst,
    sofa::gpu::opencl::_device_pointer d_Src,
    unsigned int n,
    unsigned int size
)
{
    DEBUG_TEXT("scanExclusiveLocal2");
    BARRIER(d_Dst,__FILE__,__LINE__);

    size_t localWorkSize, globalWorkSize;

    unsigned int elements = n * size;
    ckScanExclusiveLocal2->setArg<_device_pointer>(0,&d_Buffer);
    ckScanExclusiveLocal2->setArg<_device_pointer>(1,&d_Dst);
    ckScanExclusiveLocal2->setArg<_device_pointer>(2,&d_Src);
    ckScanExclusiveLocal2->setArg( 3, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);
    ckScanExclusiveLocal2->setArg<unsigned int>(4,&elements);
    ckScanExclusiveLocal2->setArg<unsigned int>(5,&size);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);

    ckScanExclusiveLocal2->execute(0, 1, NULL, &globalWorkSize, &localWorkSize);

    BARRIER(d_Dst,__FILE__,__LINE__);
    DEBUG_TEXT("~scanExclusiveLocal2");
}

void Scan::uniformUpdate(
    sofa::gpu::opencl::_device_pointer d_Dst,
    sofa::gpu::opencl::_device_pointer d_Buffer,
    unsigned int n
)
{
    DEBUG_TEXT("uniformUpdate");
    BARRIER(d_Dst,__FILE__,__LINE__);
    size_t localWorkSize, globalWorkSize;

    ckUniformUpdate->setArg<cl_mem>(0,&d_Dst.m);
    ckUniformUpdate->setArg<cl_mem>(1,&d_Buffer.m);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = n * WORKGROUP_SIZE;

    ckUniformUpdate->execute(0, 1, NULL, &globalWorkSize, &localWorkSize);

    BARRIER(d_Dst,__FILE__,__LINE__);
    DEBUG_TEXT("~uniformUpdate");
}
