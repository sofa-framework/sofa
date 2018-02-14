/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <oclUtils.h>
#include "RadixSort.h"

#define MAX_GPU_COUNT 8

int keybits = 32; // bit size of uint

// forward declarations
void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);
bool verifySortUint(unsigned int *keysSorted,
        unsigned int *valuesSorted,
        unsigned int *keysUnsorted,
        unsigned int len);

int main(int argc, const char **argv)
{
    cl_platform_id cpPlatform;                      // OpenCL platform
    cl_uint nDevice;                                // OpenCL device count
    cl_device_id* cdDevices;                        // OpenCL device list
    cl_context cxGPUContext;                        // OpenCL context
    cl_command_queue cqCommandQueue[MAX_GPU_COUNT]; // OpenCL command que
    cl_int ciErrNum;

    shrSetLogFileName ("oclRadixSort.txt");
    shrLog("%s starting...\n\n", argv[0]);

    shrLog("clGetPlatformID...\n");
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("clGetDeviceIDs...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &nDevice);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(nDevice * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, nDevice, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("clCreateContext...\n");
    cxGPUContext = clCreateContext(0, nDevice, cdDevices, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("Create command queue...\n\n");
    int id_device;
    if(shrGetCmdLineArgumenti(argc, argv, "device", &id_device)) // Set up command queue(s) for GPU specified on the command line
    {
        // get & log device index # and name
        cl_device_id cdDevice = cdDevices[id_device];

        // create a command que
        cqCommandQueue[0] = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
        oclPrintDevInfo(LOGBOTH, cdDevice);
        nDevice = 1;
    }
    else
    {
        // create command queues for all available devices
        for (cl_uint i = 0; i < nDevice; i++)
        {
            cqCommandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[i], 0, &ciErrNum);
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
        }
        for (cl_uint i = 0; i < nDevice; i++) oclPrintDevInfo(LOGBOTH, cdDevices[i]);
    }

    int ctaSize;
    if (!shrGetCmdLineArgumenti(argc, argv, "work-group-size", &ctaSize))
    {
        ctaSize = 128;
    }

    shrLog("Running Radix Sort on %d GPU(s) ...\n\n", nDevice);

    unsigned int numElements = 1048576;//128*128*128*2;

    // Alloc and init some data on the host, then alloc and init GPU buffer
    unsigned int **h_keys       = (unsigned int**)malloc(nDevice * sizeof(unsigned int*));
    unsigned int **h_keysSorted = (unsigned int**)malloc(nDevice * sizeof(unsigned int*));
    cl_mem       *d_keys        = (cl_mem*       )malloc(nDevice * sizeof(cl_mem));
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        h_keys[iDevice]       = (unsigned int*)malloc(numElements * sizeof(unsigned int));
        h_keysSorted[iDevice] = (unsigned int*)malloc(numElements * sizeof(unsigned int));
        makeRandomUintVector(h_keys[iDevice], numElements, keybits);

        d_keys[iDevice] = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                sizeof(unsigned int) * numElements, NULL, &ciErrNum);
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[iDevice], d_keys[iDevice], CL_TRUE, 0,
                sizeof(unsigned int) * numElements, h_keys[iDevice], 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    }

    // instantiate RadixSort objects
    RadixSort **radixSort = (RadixSort**)malloc(nDevice * sizeof(RadixSort*));
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        radixSort[iDevice] = new RadixSort(cxGPUContext, cqCommandQueue[iDevice], numElements, argv[0], ctaSize, true);
    }

#ifdef GPU_PROFILING
    int numIterations = 30;
    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
            {
                clFinish(cqCommandQueue[iDevice]);
            }
            shrDeltaT(1);
        }
#endif
        for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
        {
            radixSort[iDevice]->sort(d_keys[iDevice], 0, numElements, keybits);
        }
#ifdef GPU_PROFILING
    }
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clFinish(cqCommandQueue[iDevice]);
    }
    double gpuTime = shrDeltaT(1)/(double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclRadixSort, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %d, Workgroup = %d\n",
            (1.0e-6 * (double)(nDevice * numElements)/gpuTime), gpuTime, nDevice * numElements, nDevice, ctaSize);
#endif

    // copy sorted keys to CPU
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clEnqueueReadBuffer(cqCommandQueue[iDevice], d_keys[iDevice], CL_TRUE, 0, sizeof(unsigned int) * numElements,
                h_keysSorted[iDevice], 0, NULL, NULL);
    }

    // Check results
    bool passed = true;
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        passed &= verifySortUint(h_keysSorted[iDevice], NULL, h_keys[iDevice], numElements);
    }
    shrLog("\n%s\n\n", passed ? "PASSED" : "FAILED");

    // cleanup allocs
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clReleaseMemObject(d_keys[iDevice]);
        free(h_keys[iDevice]);
        free(h_keysSorted[iDevice]);
        delete radixSort[iDevice];
    }
    free(radixSort);
    free(h_keys);
    free(h_keysSorted);

    // remaining cleanup and exit
    free(cdDevices);
    for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clReleaseCommandQueue(cqCommandQueue[iDevice]);
    }
    clReleaseContext(cxGPUContext);
    shrEXIT(argc, argv);
}

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)
    {
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask);
    }
}

// assumes the values were initially indices into the array, for simplicity of
// checking correct order of values
bool verifySortUint(unsigned int *keysSorted,
        unsigned int *valuesSorted,
        unsigned int *keysUnsorted,
        unsigned int len)
{
    bool passed = true;
    for(unsigned int i=0; i<len-1; ++i)
    {
        if( (keysSorted[i])>(keysSorted[i+1]) )
        {
            shrLog("Unordered key[%d]: %d > key[%d]: %d\n", i, keysSorted[i], i+1, keysSorted[i+1]);
            passed = false;
            break;
        }
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
            {
                shrLog("Incorrectly sorted value[%u] (%u): %u != %u\n",
                        i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                passed = false;
                break;
            }
        }
    }

    return passed;
}
