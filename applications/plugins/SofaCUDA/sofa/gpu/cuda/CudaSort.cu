/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
//#include "CudaSort.h"
#include "mycuda.h"

#include <cuda.h>

#if defined(SOFA_GPU_CUDPP)
#include <cudpp.h>
#include <cudpp_plan.h>
#include <cudpp_plan_manager.h>
#include <cudpp_radixsort.h>
#endif

#if defined(SOFA_GPU_THRUST)
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

#if !defined(SOFA_GPU_CUDPP) && !defined(SOFA_GPU_THRUST)
#ifdef _MSC_VER
#pragma message( __FILE__ " : Warning: CUDA: please define either SOFA_GPU_CUDPP or SOFA_GPU_THRUST to activate sorting on GPU")
#else
#warning CUDA: please define either SOFA_GPU_CUDPP or SOFA_GPU_THRUST to activate sorting on GPU
#endif
#endif

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C" {

    bool SOFA_GPU_CUDA_API CudaSortGPUAvailable(unsigned int size, bool withData = true);
    bool SOFA_GPU_CUDA_API CudaSortGPU(void* keys, void* data, unsigned int size, int bits);

} // "C"

#if defined(SOFA_GPU_CUDPP)

CUDPPHandle cudppHandleSort[2];
unsigned int cudppHandleSortMaxElements[2] = { 0, 0 };
bool cudppFailed = false;

bool CudaSortCUDPPAvailable(unsigned int numElements, bool withData)
{
    if (cudppFailed) return false;
    int plan = (withData) ? 1 : 0;
    if (numElements > cudppHandleSortMaxElements[plan])
    {
        if (cudppHandleSortMaxElements[plan] > 0)
        {
            cudppDestroyPlan(cudppHandleSort[plan]);
            cudppHandleSortMaxElements[plan] = (((cudppHandleSortMaxElements[plan]>>10)+1)<<10); // increase size to at least the next multiple of 1024
        }
        if (numElements > cudppHandleSortMaxElements[plan])
            cudppHandleSortMaxElements[plan] = numElements;
//            if (cudppHandleSortMaxElements[plan] < (1<<18))
//                cudppHandleSortMaxElements[plan] = (1<<18);
        cudppHandleSortMaxElements[plan] = ((cudppHandleSortMaxElements[plan] + 255) & ~255);

        mycudaPrintf("CudaSort: Creating CUDPP RadixSort Plan for %d elements.\n", cudppHandleSortMaxElements[plan]);
        CUDPPConfiguration config;
        config.algorithm = CUDPP_SORT_RADIX;
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_UINT;
        config.options = withData ? CUDPP_OPTION_KEY_VALUE_PAIRS : CUDPP_OPTION_KEYS_ONLY;
        if (cudppPlan(&cudppHandleSort[plan], config, cudppHandleSortMaxElements[plan], 1, 0) != CUDPP_SUCCESS)
        {
            mycudaPrintf("CudaSort: ERROR creating CUDPP RadixSort Plan for %d elements.\n", cudppHandleSortMaxElements[plan]);
            cudppHandleSortMaxElements[plan] = 0;
            cudppDestroyPlan(cudppHandleSort[plan]);
            cudppFailed = true;
            return false;
        }
    }
    return true;
}

bool CudaSortCUDPP(void * d_keys, void * d_values, unsigned int numElements, int keybits = 32)
{
    bool withData = (d_values != NULL);
    if (!CudaSortCUDPPAvailable(numElements, withData))
        return false;
    int plan = (withData) ? 1 : 0;
    if (cudppSort(cudppHandleSort[plan],d_keys,d_values,keybits,numElements) != CUDPP_SUCCESS)
        return false;
    return true;
}
#endif

#if defined(SOFA_GPU_THRUST)

unsigned int thrustSortMaxElements = 0;

bool CudaSortTHRUSTAvailable(unsigned int numElements, bool /*withData*/)
{
    if (numElements > thrustSortMaxElements)
    {
        if (numElements > thrustSortMaxElements)
            thrustSortMaxElements = numElements;
//            if (thrustSortMaxElements < (1<<18))
//                thrustSortMaxElements = (1<<18);
        thrustSortMaxElements = ((thrustSortMaxElements + 255) & ~255);

//        mycudaPrintf("CudaSort: Using THRUST to sort up to %d elements.\n", thrustSortMaxElements);
    }
    return true;
}

bool CudaSortTHRUST(void* keys, void* data, unsigned int size, int /*bits*/)
{
    if (!CudaSortTHRUSTAvailable(size, (data != NULL)))
        return false;
    thrust::device_ptr<unsigned int> d_keys ( (unsigned int*) keys );
    if (data)
    {
        thrust::device_ptr<unsigned int> d_data ( (unsigned int*) data );
        thrust::sort_by_key(d_keys, d_keys+size, d_data);
    }
    else
    {
        thrust::sort(d_keys, d_keys+size);
    }
    return true;
}

#endif

enum SortImplType
{
    SORTDEFAULT = 0,
#if defined(SOFA_GPU_CUDPP)
    SORT_CUDPP,
#endif
#if defined(SOFA_GPU_THRUST)
    SORT_THRUST,
#endif
    SORT_UNKNOWN
};

SortImplType CudaSortImpl()
{
    static bool done = false;
    static SortImplType impl = SORTDEFAULT;
    if (!done)
    {
        const char* str = mygetenv("CUDA_SORT");
        if (!str || !*str)
            impl = SORTDEFAULT;
        else if ((str[0] == 'D' || str[0] == 'd') && (str[1] == 'E' || str[1] == 'e'))
            impl = SORTDEFAULT;
#if defined(SOFA_GPU_CUDPP)
        else if ((str[0] == 'C' || str[0] == 'c') && (str[1] == 'U' || str[1] == 'u'))
            impl = SORT_CUDPP;
#endif
#if defined(SOFA_GPU_THRUST)
        else if ((str[0] == 'T' || str[0] == 't') && (str[1] == 'H' || str[1] == 'h'))
            impl = SORT_THRUST;
#endif
        else
            impl = SORT_UNKNOWN;
        done = true;
    }
    return impl;
}

bool CudaSortGPUAvailable(unsigned int size, bool withData)
{
    SortImplType impl = CudaSortImpl();
    switch(impl)
    {
    case SORTDEFAULT: // alias for the first active implementation
#if defined(SOFA_GPU_CUDPP)
    case SORT_CUDPP:
        if (CudaSortCUDPPAvailable(size, withData))
            return true;
        if (impl != SORTDEFAULT)
            break;
#endif
#if defined(SOFA_GPU_THRUST)
    case SORT_THRUST:
        if (CudaSortTHRUSTAvailable(size, withData))
            return true;
        if (impl != SORTDEFAULT)
            break;
#endif
    case SORT_UNKNOWN:
        return false;
    }
    return false;
}

bool CudaSortGPU(void* keys, void* data, unsigned int size, int bits)
{
    SortImplType impl = CudaSortImpl();
    switch(impl)
    {
    case SORTDEFAULT: // alias for the first active implementation
#if defined(SOFA_GPU_CUDPP)
    case SORT_CUDPP:
        if (CudaSortCUDPP(keys, data, size, bits))
            return true;
        if (impl != SORTDEFAULT)
            break;
#endif
#if defined(SOFA_GPU_THRUST)
    case SORT_THRUST:
        if (CudaSortTHRUST(keys, data, size, bits))
            return true;
        if (impl != SORTDEFAULT)
            break;
#endif
    case SORT_UNKNOWN:
        return false;
    }
    return false;
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
