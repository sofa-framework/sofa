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
#ifndef SOFA_GPU_CUDA_CUDASCAN_H
#define SOFA_GPU_CUDA_CUDASCAN_H

#include <sofa/gpu/cuda/mycuda.h>

#if defined(__cplusplus)

#include <sofa/gpu/cuda/CudaTypes.h>

#include <functional>
#include <numeric> // for std::partial_sum

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

    extern bool SOFA_GPU_CUDA_API CudaScanGPUAvailable(unsigned int size, ScanType type);
    extern bool SOFA_GPU_CUDA_API CudaScanGPU(const void* input, void* output, unsigned int size, ScanType type);

} // "C"

#if defined(__cplusplus)

template<class TData>
void CudaScanCPU(const TData* input, TData* output, unsigned int size, ScanType type)
{
    switch(type)
    {
    case SCAN_INCLUSIVE:
        std::partial_sum(input,input+size, output);
        break;
    case SCAN_EXCLUSIVE:
        output[0] = 0;
        std::partial_sum(input, input+(size-1), output+1);
        break;
    }
}

#endif

static inline void CudaScanPrepare(unsigned int size, ScanType type)
{
    if (!CudaScanGPUAvailable(size, type))
        std::cerr << "CUDA: GPU scan implementation not available (size="<<size<<")" << std::endl;
}

template<class TData>
static inline void CudaScan(const CudaVector<TData>* input, unsigned int input0, CudaVector<TData>* output, unsigned int output0, unsigned int size, ScanType type, bool forceCPU = false)
{
    bool withCPU = forceCPU;
    if (!withCPU && !CudaScanGPU(input->deviceReadAt(input0), output->deviceWriteAt(output0), size, type))
        withCPU = true;
    if (withCPU)
    {
        CudaScanCPU(input->hostReadAt(input0), output->hostWriteAt(output0), size, type);
    }
}

template<class TData>
static inline void CudaScan(const CudaVector<TData>* input, CudaVector<TData>* output, unsigned int size, ScanType type, bool forceCPU = false)
{
    CudaScan(input, 0, output, 0, size, type, forceCPU);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
