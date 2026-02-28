/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#if defined(SOFA_GPU_THRUST)
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

#if !defined(SOFA_GPU_THRUST)
#ifdef _MSC_VER
#pragma message( __FILE__ " : Warning: CUDA: please define SOFA_GPU_THRUST to activate sorting on GPU")
#else
#warning CUDA: please define SOFA_GPU_THRUST to activate sorting on GPU
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
    const thrust::device_ptr<unsigned int> d_keys ( (unsigned int*) keys );
    if (data)
    {
        const thrust::device_ptr<unsigned int> d_data ( (unsigned int*) data );
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
