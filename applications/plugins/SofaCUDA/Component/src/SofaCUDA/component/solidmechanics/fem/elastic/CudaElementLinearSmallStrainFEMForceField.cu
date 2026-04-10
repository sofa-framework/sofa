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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include "CudaElementFEMKernelUtils.cuh"

namespace sofa::gpu::cuda
{

/**
 * Kernel for addForce: f = -K * (x - x0)
 */
template<typename T, int NNodes, int Dim>
__global__ void ElementLinearSmallStrainFEMForceField_computeForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const T* __restrict__ stiffness,
    const T* __restrict__ x,
    const T* __restrict__ x0,
    T* __restrict__ eforce)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather displacement (x - x0)
    T disp[NNodes * Dim];
    gatherElementDisplacement<T, NNodes, Dim>(elements, nbElem, elemId, x, x0, disp);

    // Multiply by stiffness matrix
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];
    symBlockMatMul<T, NNodes, Dim>(K, disp, edf);

    // Write negated force
    T* out = eforce + elemId * NNodes * Dim;
    writeForce<T, NNodes, Dim>(edf, out, T(-1));
}

/**
 * Kernel for addDForce: df = -kFactor * K * dx
 */
template<typename T, int NNodes, int Dim>
__global__ void ElementLinearSmallStrainFEMForceField_computeDForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const T* __restrict__ stiffness,
    const T* __restrict__ dx,
    T* __restrict__ eforce,
    T kFactor)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather displacement increment
    T edx[NNodes * Dim];
    gatherElementData<T, NNodes, Dim>(elements, nbElem, elemId, dx, edx);

    // Multiply by stiffness matrix
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];
    symBlockMatMul<T, NNodes, Dim>(K, edx, edf);

    // Write scaled negated force
    T* out = eforce + elemId * NNodes * Dim;
    writeForce<T, NNodes, Dim>(edf, out, -kFactor);
}

// ===================== Launch functions =====================

template<typename T, int NNodes, int Dim>
void ElementLinearSmallStrainFEMForceFieldCuda_addForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* stiffness,
    const void* x,
    const void* x0,
    void* f,
    void* eforce,
    const void* velems)
{
    const int computeThreads = 64;
    int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementLinearSmallStrainFEMForceField_computeForce_kernel<T, NNodes, Dim>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)stiffness,
            (const T*)x,
            (const T*)x0,
            (T*)eforce);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_computeForce_kernel");

    const int gatherThreads = 256;
    numBlocks = (nbVertex + gatherThreads - 1) / gatherThreads;
    ElementFEM_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementFEM_gatherForce_kernel");
}

template<typename T, int NNodes, int Dim>
void ElementLinearSmallStrainFEMForceFieldCuda_addDForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* stiffness,
    const void* dx,
    void* df,
    void* eforce,
    const void* velems,
    T kFactor)
{
    const int computeThreads = 64;
    int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementLinearSmallStrainFEMForceField_computeDForce_kernel<T, NNodes, Dim>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)stiffness,
            (const T*)dx,
            (T*)eforce,
            kFactor);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_computeDForce_kernel");

    const int gatherThreads = 256;
    numBlocks = (nbVertex + gatherThreads - 1) / gatherThreads;
    ElementFEM_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)df);
    mycudaDebugError("ElementFEM_gatherForce_kernel");
}

// ===================== Explicit template instantiations =====================

#define INSTANTIATE_LINEAR(T, NNodes) \
    template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<T, NNodes, 3>( \
        unsigned int, unsigned int, unsigned int, const void*, const void*, \
        const void*, const void*, void*, void*, const void*); \
    template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<T, NNodes, 3>( \
        unsigned int, unsigned int, unsigned int, const void*, const void*, \
        const void*, void*, void*, const void*, T);

INSTANTIATE_LINEAR(float, 2)
INSTANTIATE_LINEAR(float, 3)
INSTANTIATE_LINEAR(float, 4)
INSTANTIATE_LINEAR(float, 8)

INSTANTIATE_LINEAR(double, 2)
INSTANTIATE_LINEAR(double, 3)
INSTANTIATE_LINEAR(double, 4)
INSTANTIATE_LINEAR(double, 8)

#undef INSTANTIATE_LINEAR

} // namespace sofa::gpu::cuda
