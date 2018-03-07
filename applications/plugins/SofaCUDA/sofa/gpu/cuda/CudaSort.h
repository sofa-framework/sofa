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
#ifndef SOFA_GPU_CUDA_CUDASORT_H
#define SOFA_GPU_CUDA_CUDASORT_H

#include <sofa/gpu/cuda/mycuda.h>

#if defined(__cplusplus)

#include <sofa/gpu/cuda/CudaTypes.h>

#include <algorithm> // for std::sort
#include <vector>

namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C" {

    extern bool SOFA_GPU_CUDA_API CudaSortGPUAvailable(unsigned int size, bool withData = true);
    extern bool SOFA_GPU_CUDA_API CudaSortGPU(void* keys, void* data, unsigned int size, int bits);

} // "C"

#if defined(__cplusplus)

struct compare_pair_first
{
    template<class T1, class T2>
    bool operator()(const std::pair<T1,T2>& a, const std::pair<T1,T2>& b) const
    {
        return a.first < b.first;
    }
};

template<class TKey, class TData>
void CudaSortCPU(TKey* keys, TData* data, unsigned int size, int /*bits*/)
{
    if (data)
    {
        std::vector< std::pair<TKey,TData> > cpusort;
        cpusort.resize(size);
        for (unsigned int i=0; i<size; ++i)
        {
            cpusort[i].first = keys[i];
            cpusort[i].second = data[i];
        }
        std::sort(cpusort.begin(),cpusort.end(),compare_pair_first());

        for (unsigned int i=0; i<size; ++i)
        {
            keys[i] = cpusort[i].first;
            data[i] = cpusort[i].second;
        }
    }
    else
    {
        std::sort(keys,keys+size);
    }
}

#endif

static inline void CudaSortPrepare(unsigned int size, bool withData = true)
{
    if (!CudaSortGPUAvailable(size, withData))
        std::cerr << "CUDA: GPU sort implementation not available (size="<<size<<")" << std::endl;
}

template<class TKey, class TData>
static inline void CudaSort(CudaVector<TKey>* keys, unsigned int key0, CudaVector<TData>* data, unsigned int data0, unsigned int size, int bits = 32, bool forceCPU = false)
{
    bool withData = (data && !data->empty());
    bool withCPU = forceCPU;
    //if (!withCPU && !CudaSortGPUAvailable(size, withData))
    //    withCPU = true;
    if (!withCPU && !CudaSortGPU(keys->deviceWriteAt(key0), (withData ? data->deviceWriteAt(data0) : NULL), size, bits))
        withCPU = true;
    if (withCPU)
    {
        CudaSortCPU(keys->hostWriteAt(key0), (withData ? data->hostWriteAt(data0) : NULL), size, bits);
    }
}

template<class TKey, class TData>
static inline void CudaSort(CudaVector<TKey>* keys, CudaVector<TData>* data, unsigned int size, int bits = 32, bool forceCPU = false)
{
    CudaSort(keys, 0, data, 0, size, bits, forceCPU);
}



#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
