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
//#include "CudaScan.h"
#include "mycuda.h"

#include <cuda.h>

#if defined(SOFA_GPU_CUDPP)
#include <cudpp.h>
#include <cudpp_plan.h>
#include <cudpp_plan_manager.h>
#include <cudpp_scan.h>
#endif

#if defined(SOFA_GPU_THRUST)
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#endif

//#if !defined(SOFA_GPU_CUDPP) && !defined(SOFA_GPU_THRUST)
//#warning CUDA: please define either SOFA_GPU_CUDPP or SOFA_GPU_THRUST to activate scan on GPU
//#endif

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

enum ScanType
{
    SCAN_INCLUSIVE = 0,
    SCAN_EXCLUSIVE = 1,
};

extern "C" {

    bool SOFA_GPU_CUDA_API CudaScanGPUAvailable(unsigned int size, ScanType type);
    bool SOFA_GPU_CUDA_API CudaScanGPU(const void* input, void* output, unsigned int size, ScanType type);

} // "C"

void* sofaScanTmpDev = NULL;
unsigned int sofaScanMaxElements = 0;

#define SCAN_THREAD 64

int CudaScan_TempStorage(int sz)
{
    return (sz+SCAN_THREAD-1)/SCAN_THREAD;
}

bool CudaScanSOFAAvailable(unsigned int size, ScanType type)
{
    if (size > sofaScanMaxElements)
    {
        if (sofaScanMaxElements > 0)
        {
            mycudaFree(sofaScanTmpDev);
            sofaScanTmpDev = NULL;
            sofaScanMaxElements = (((sofaScanMaxElements*5>>12)+1)<<10); // increase size to at least 25% and to the next multiple of 1024
        }
        if (size > sofaScanMaxElements)
            sofaScanMaxElements = size;
//            if (sofaScanMaxElements < (1<<18))
//                sofaScanMaxElements = (1<<18);
        sofaScanMaxElements = ((sofaScanMaxElements + 255) & ~255);

        mycudaPrintf("CudaScan: Using SOFA Scan for %d elements.\n", sofaScanMaxElements);
        int tmpsize = CudaScan_TempStorage(sofaScanMaxElements);
        mycudaMalloc(&sofaScanTmpDev, tmpsize*sizeof(unsigned int));
    }
    return true;
}

template<class TInt, bool exclusive>
__global__ void CudaScan_count_kernel(TInt* tmp,const TInt * count,unsigned int sz)
{
    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;

    __shared__ TInt stemp[SCAN_THREAD];
    volatile TInt * temp = stemp;

    unsigned int read = bx*SCAN_THREAD;

    if (read+tx<(exclusive ? sz-1 : sz)) temp[tx] = count[read+tx];
    else temp[tx] = 0;

#if SCAN_THREAD > 128
    __syncthreads();
    if (tx < 128) temp[tx]+=temp[tx+128];
#endif
#if SCAN_THREAD > 64
    __syncthreads();
    if (tx < 64) temp[tx]+=temp[tx+64];
#endif
#if SCAN_THREAD > 32
    __syncthreads();
    if (tx < 32) temp[tx]+=temp[tx+32];
#endif
#if SCAN_THREAD > 16
    if (tx < 16) temp[tx]+=temp[tx+16];
#endif
#if SCAN_THREAD > 8
    if (tx < 8) temp[tx]+=temp[tx+8];
#endif
#if SCAN_THREAD > 4
    if (tx < 4) temp[tx]+=temp[tx+4];
#endif
#if SCAN_THREAD > 2
    if (tx < 2) temp[tx]+=temp[tx+2];
#endif
#if SCAN_THREAD > 1
    if (tx < 1) temp[tx]+=temp[tx+1];
#endif

    if (tx==0) tmp[bx] = temp[0];
}

template<class TInt, bool exclusive>
__global__ void CudaScan_main_kernel(TInt * tmp,TInt * start,const TInt * count, unsigned int sz, unsigned int nbb)
{
    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;

    __shared__ TInt stemp[2*SCAN_THREAD];
    //__shared__ TInt temp0;
    volatile TInt * temp = stemp;

    unsigned int write = bx*SCAN_THREAD;
    if (write>=sz) return;

    temp[tx] = 0.0;
    for (unsigned int b=0; (b+tx<bx); b+=SCAN_THREAD)
    {
        temp[tx] += tmp[b+tx];
    }

#if SCAN_THREAD > 128
    __syncthreads();
    if (tx < 128) temp[tx]+=temp[tx+128];
#endif
#if SCAN_THREAD > 64
    __syncthreads();
    if (tx < 64) temp[tx]+=temp[tx+64];
#endif
#if SCAN_THREAD > 32
    __syncthreads();
    if (tx < 32) temp[tx]+=temp[tx+32];
#endif
#if SCAN_THREAD > 16
    if (tx < 16) temp[tx]+=temp[tx+16];
#endif
#if SCAN_THREAD > 8
    if (tx < 8) temp[tx]+=temp[tx+8];
#endif
#if SCAN_THREAD > 4
    if (tx < 4) temp[tx]+=temp[tx+4];
#endif
#if SCAN_THREAD > 2
    if (tx < 2) temp[tx]+=temp[tx+2];
#endif
#if SCAN_THREAD > 1
    if (tx < 1) temp[tx]+=temp[tx+1];
#endif

    int tval;
    if (write+tx<(exclusive ? sz-1 : sz)) tval = count[write+tx];
    else tval = 0;

    if (tx==0) tval += temp[0];

    temp[SCAN_THREAD+tx] = tval;

#if SCAN_THREAD > 32
    __syncthreads();
#endif

    if (tx >= 1)
        tval += temp[SCAN_THREAD+tx-1];

#if SCAN_THREAD > 2
    temp[tx] = tval;
    __syncthreads();
    if (tx >= 2)
        tval += temp[tx-2];
#endif

#if SCAN_THREAD > 4
    temp[SCAN_THREAD+tx] = tval;
    __syncthreads();
    if (tx >= 4)
        tval += temp[SCAN_THREAD+tx-4];
#endif

#if SCAN_THREAD > 8
    temp[tx] = tval;
    __syncthreads();
    if (tx >= 8)
        tval += temp[tx-8];
#endif

#if SCAN_THREAD > 16
    temp[SCAN_THREAD+tx] = tval;
    __syncthreads();
    if (tx >= 16)
        tval += temp[SCAN_THREAD+tx-16];
#endif

#if SCAN_THREAD > 32
    temp[tx] = tval;
    __syncthreads();
    if (tx >= 32)
        tval += temp[tx-32];
#endif

#if SCAN_THREAD > 64
    temp[SCAN_THREAD+tx] = tval;
    __syncthreads();
    if (tx >= 64)
        tval += temp[SCAN_THREAD+tx-64];
#endif

#if SCAN_THREAD > 128
    temp[tx] = tval;
    __syncthreads();
    if (tx >= 128)
        tval += temp[tx-128];
#endif

    unsigned int out = (exclusive ? write+tx+1 : write+tx);
    if (out<sz) start[out] = tval;
    if (exclusive && (tx==0) && (bx==0)) start[0] = 0;
}

bool CudaScanSOFA(const void* input, void* output, unsigned int size, ScanType type)
{
    if (!CudaScanSOFAAvailable(size, type))
        return false;
    unsigned int nbb = (size+SCAN_THREAD-1)/SCAN_THREAD;
    dim3 threads(SCAN_THREAD,1);
    dim3 grid(nbb,1);
    void* tmp = sofaScanTmpDev;
    switch(type)
    {
    case SCAN_INCLUSIVE:
    {
        {CudaScan_count_kernel<unsigned int, false><<< grid, threads>>>((unsigned int*) tmp,(const unsigned int *) input,size); mycudaDebugError("CudaScan_count_kernel<unsigned int, false>");}
        {CudaScan_main_kernel<unsigned int, false><<< grid, threads>>>((unsigned int*) tmp,(unsigned int*) output,(const unsigned int *) input,size,nbb); mycudaDebugError("CudaScan_main_kernel<unsigned int, false>");}
        return true;
    }
    case SCAN_EXCLUSIVE:
    {CudaScan_count_kernel<unsigned int, true><<< grid, threads>>>((unsigned int*) tmp,(const unsigned int *) input,size); mycudaDebugError("CudaScan_count_kernel<unsigned int, true>");}
    {CudaScan_main_kernel<unsigned int, true><<< grid, threads>>>((unsigned int*) tmp,(unsigned int*) output,(const unsigned int *) input,size,nbb); mycudaDebugError("CudaScan_main_kernel<unsigned int, true>");}
    return true;
    default:
        return false;
    }
}


#if defined(SOFA_GPU_CUDPP)

CUDPPHandle cudppHandleScan[2];
unsigned int cudppHandleScanMaxElements[2] = { 0, 0 };
bool cudppScanFailed = false;

bool CudaScanCUDPPAvailable(unsigned int size, ScanType type)
{
    if (cudppScanFailed) return false;
    int plan = (type == SCAN_INCLUSIVE ? 0 : 1);
    if (size > cudppHandleScanMaxElements[plan])
    {
        if (cudppHandleScanMaxElements[plan] > 0)
        {
            cudppDestroyPlan(cudppHandleScan[plan]);
            cudppHandleScanMaxElements[plan] = (((cudppHandleScanMaxElements[plan]>>10)+1)<<10); // increase size to at least the next multiple of 1024
        }
        if (size > cudppHandleScanMaxElements[plan])
            cudppHandleScanMaxElements[plan] = size;
//            if (cudppHandleScanMaxElements[plan] < (1<<18))
//                cudppHandleScanMaxElements[plan] = (1<<18);
        cudppHandleScanMaxElements[plan] = ((cudppHandleScanMaxElements[plan] + 255) & ~255);

        mycudaPrintf("CudaScan: Creating CUDPP %s Scan Plan for %d elements.\n", (type == SCAN_INCLUSIVE ? "Inclusive" : "Exclusive"), cudppHandleScanMaxElements[plan]);
        CUDPPConfiguration config;
        config.algorithm = CUDPP_SCAN;
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_UINT;
        config.options = (type == SCAN_INCLUSIVE ? CUDPP_OPTION_INCLUSIVE : CUDPP_OPTION_EXCLUSIVE);

        if (cudppPlan(&cudppHandleScan[plan], config, cudppHandleScanMaxElements[plan], 1, 0) != CUDPP_SUCCESS)
        {
            mycudaPrintf("CudaScan: ERROR creating CUDPP %s Scan Plan for %d elements.\n", (type == SCAN_INCLUSIVE ? "Inclusive" : "Exclusive"), cudppHandleScanMaxElements[plan]);
            cudppHandleScanMaxElements[plan] = 0;
            cudppDestroyPlan(cudppHandleScan[plan]);
            cudppScanFailed = true;
            return false;
        }
    }
    return true;
}

bool CudaScanCUDPP(const void * d_input, void * d_output, unsigned int size, ScanType type)
{
    if (!CudaScanCUDPPAvailable(size, type))
        return false;
    int plan = (type == SCAN_INCLUSIVE ? 0 : 1);
    if (cudppScan(cudppHandleScan[plan],d_output,d_input,size) != CUDPP_SUCCESS)
        return false;
    return true;
}
#endif

#if defined(SOFA_GPU_THRUST)

unsigned int thrustScanMaxElements = 0;

bool CudaScanTHRUSTAvailable(unsigned int size, bool /*withData*/)
{
    if (size > thrustScanMaxElements)
    {
        if (size > thrustScanMaxElements)
            thrustScanMaxElements = size;
//            if (thrustScanMaxElements < (1<<18))
//                thrustScanMaxElements = (1<<18);
        thrustScanMaxElements = ((thrustScanMaxElements + 255) & ~255);

//        mycudaPrintf("CudaScan: Using THRUST to scan up to %d elements.\n", thrustScanMaxElements);
    }
    return true;
}

bool CudaScanTHRUST(const void* input, void* output, unsigned int size, ScanType type)
{
    if (!CudaScanTHRUSTAvailable(size, type))
        return false;
    thrust::device_ptr<unsigned int> d_input ( (unsigned int*) input );
    thrust::device_ptr<unsigned int> d_output ( (unsigned int*) output );
    switch(type)
    {
    case SCAN_INCLUSIVE:
        thrust::inclusive_scan(d_input, d_input+size, d_output);
        return true;
    case SCAN_EXCLUSIVE:
        thrust::exclusive_scan(d_input, d_input+size, d_output);
        return true;
    default:
        return false;
    }
}

#endif

enum ScanImplType
{
    SCANDEFAULT = 0,
    SCAN_SOFA,
#if defined(SOFA_GPU_CUDPP)
    SCAN_CUDPP,
#endif
#if defined(SOFA_GPU_THRUST)
    SCAN_THRUST,
#endif
    SCAN_UNKNOWN
};

ScanImplType CudaScanImpl()
{
    static bool done = false;
    static ScanImplType impl = SCANDEFAULT;
    if (!done)
    {
        const char* str = mygetenv("CUDA_SCAN");
        if (!str || !*str)
            impl = SCANDEFAULT;
        else if ((str[0] == 'D' || str[0] == 'd') && (str[1] == 'E' || str[1] == 'e'))
            impl = SCANDEFAULT;
        else if ((str[0] == 'S' || str[0] == 's') && (str[1] == 'O' || str[1] == 'o'))
            impl = SCAN_SOFA;
#if defined(SOFA_GPU_CUDPP)
        else if ((str[0] == 'C' || str[0] == 'c') && (str[1] == 'U' || str[1] == 'u'))
            impl = SCAN_CUDPP;
#endif
#if defined(SOFA_GPU_THRUST)
        else if ((str[0] == 'T' || str[0] == 't') && (str[1] == 'H' || str[1] == 'h'))
            impl = SCAN_THRUST;
#endif
        else
            impl = SCAN_UNKNOWN;
        done = true;
    }
    return impl;
}

bool CudaScanGPUAvailable(unsigned int size, ScanType type)
{
    ScanImplType impl = CudaScanImpl();
    switch(impl)
    {
    case SCANDEFAULT: // alias for the first active implementation
#if defined(SOFA_GPU_CUDPP)
    case SCAN_CUDPP:
        if (CudaScanCUDPPAvailable(size, type))
            return true;
        if (impl != SCANDEFAULT)
            break;
#endif
#if defined(SOFA_GPU_THRUST)
    case SCAN_THRUST:
        if (CudaScanTHRUSTAvailable(size, type))
            return true;
        if (impl != SCANDEFAULT)
            break;
#endif
    case SCAN_UNKNOWN:
        return false;
    }
    return false;
}

bool CudaScanGPU(const void* input, void* output, unsigned int size, ScanType type)
{
    ScanImplType impl = CudaScanImpl();
    switch(impl)
    {
    case SCANDEFAULT: // alias for the first active implementation
#if defined(SOFA_GPU_CUDPP)
    case SCAN_CUDPP:
        if (CudaScanCUDPP(input, output, size, type))
            return true;
        if (impl != SCANDEFAULT)
            break;
#endif
#if defined(SOFA_GPU_THRUST)
    case SCAN_THRUST:
        if (CudaScanTHRUST(input, output, size, type))
            return true;
        if (impl != SCANDEFAULT)
            break;
#endif
    case SCAN_UNKNOWN:
        return false;
    }
    return false;
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
