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
 * Combined kernel: compute rotations AND per-element forces in one pass.
 */
template<typename T, int NNodes, int Dim>
__global__ void ElementCorotationalFEMForceField_computeRotationsAndForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const T* __restrict__ initRotTransposed,
    const T* __restrict__ stiffness,
    const T* __restrict__ x,
    const T* __restrict__ x0,
    T* __restrict__ rotationsOut,
    T* __restrict__ eforce)
{
    static_assert(Dim == 3, "Corotational rotation computation requires Dim == 3");
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather element positions
    T ex[NNodes * Dim], ex0[NNodes * Dim];
    gatherElementData<T, NNodes, Dim>(elements, nbElem, elemId, x, ex);
    gatherElementData<T, NNodes, Dim>(elements, nbElem, elemId, x0, ex0);

    // Compute rotation frame from current positions
    T frame[Dim * Dim];
    if constexpr (NNodes == 8)
        computeHexahedronFrame(ex, frame);
    else
        computeTriangleFrame(ex, frame);

    // R = frame^T * initRotTransposed
    const T* irt = initRotTransposed + elemId * Dim * Dim;
    T R[Dim * Dim];
    mat3TransposeMul(frame, irt, R);

    // Store rotation for later use
    T* Rout = rotationsOut + elemId * Dim * Dim;
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        Rout[i] = R[i];

    // Compute element centers
    T center[Dim], center0[Dim];
    computeElementCenter<T, NNodes, Dim>(ex, center);
    computeElementCenter<T, NNodes, Dim>(ex0, center0);

    // Compute corotational displacement
    T disp[NNodes * Dim];
    computeCorotationalDisplacement<T, NNodes, Dim>(R, ex, ex0, center, center0, disp);

    // Multiply by stiffness matrix
    T edf[NNodes * Dim];
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    symBlockMatMul<T, NNodes, Dim>(K, disp, edf);

    // Rotate forces back to global frame and negate
    T* out = eforce + elemId * NNodes * Dim;
    rotateAndWriteForce<T, NNodes, Dim>(R, edf, out, T(-1));
}

/**
 * Kernel for addForce with pre-computed rotations.
 */
template<typename T, int NNodes, int Dim>
__global__ void ElementCorotationalFEMForceField_computeForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const T* __restrict__ rotations,
    const T* __restrict__ stiffness,
    const T* __restrict__ x,
    const T* __restrict__ x0,
    T* __restrict__ eforce)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load rotation matrix
    const T* Rptr = rotations + elemId * Dim * Dim;
    T R[Dim * Dim];
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        R[i] = Rptr[i];

    // Gather element positions
    T ex[NNodes * Dim], ex0[NNodes * Dim];
    gatherElementData<T, NNodes, Dim>(elements, nbElem, elemId, x, ex);
    gatherElementData<T, NNodes, Dim>(elements, nbElem, elemId, x0, ex0);

    // Compute element centers
    T center[Dim], center0[Dim];
    computeElementCenter<T, NNodes, Dim>(ex, center);
    computeElementCenter<T, NNodes, Dim>(ex0, center0);

    // Compute corotational displacement
    T disp[NNodes * Dim];
    computeCorotationalDisplacement<T, NNodes, Dim>(R, ex, ex0, center, center0, disp);

    // Multiply by stiffness matrix
    T edf[NNodes * Dim];
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    symBlockMatMul<T, NNodes, Dim>(K, disp, edf);

    // Rotate forces back to global frame and negate
    T* out = eforce + elemId * NNodes * Dim;
    rotateAndWriteForce<T, NNodes, Dim>(R, edf, out, T(-1));
}

/**
 * Kernel for addDForce.
 */
template<typename T, int NNodes, int Dim>
__global__ void ElementCorotationalFEMForceField_computeDForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const T* __restrict__ rotations,
    const T* __restrict__ stiffness,
    const T* __restrict__ dx,
    T* __restrict__ eforce,
    T kFactor)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load rotation matrix
    const T* Rptr = rotations + elemId * Dim * Dim;
    T R[Dim * Dim];
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        R[i] = Rptr[i];

    // Gather and rotate displacement: rdx = R^T * dx
    T rdx[NNodes * Dim];
    rotateDisplacementTranspose<T, NNodes, Dim>(R, elements, nbElem, elemId, dx, rdx);

    // Multiply by stiffness matrix
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];
    symBlockMatMul<T, NNodes, Dim>(K, rdx, edf);

    // Rotate forces back to global frame and scale
    T* out = eforce + elemId * NNodes * Dim;
    rotateAndWriteForce<T, NNodes, Dim>(R, edf, out, -kFactor);
}

// ===================== Launch functions =====================

template<typename T, int NNodes, int Dim>
void ElementCorotationalFEMForceFieldCuda_addForceWithRotations(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* initRotTransposed,
    const void* stiffness,
    const void* x,
    const void* x0,
    void* f,
    void* eforce,
    void* rotationsOut,
    const void* velems)
{
    const int computeThreads = 64;
    int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeRotationsAndForce_kernel<T, NNodes, Dim>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)initRotTransposed,
            (const T*)stiffness,
            (const T*)x,
            (const T*)x0,
            (T*)rotationsOut,
            (T*)eforce);
    mycudaDebugError("ElementCorotationalFEMForceField_computeRotationsAndForce_kernel");

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
void ElementCorotationalFEMForceFieldCuda_addForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* rotations,
    const void* stiffness,
    const void* x,
    const void* x0,
    void* f,
    void* eforce,
    const void* velems)
{
    const int computeThreads = 64;
    int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeForce_kernel<T, NNodes, Dim>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)rotations,
            (const T*)stiffness,
            (const T*)x,
            (const T*)x0,
            (T*)eforce);
    mycudaDebugError("ElementCorotationalFEMForceField_computeForce_kernel");

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
void ElementCorotationalFEMForceFieldCuda_addDForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* rotations,
    const void* stiffness,
    const void* dx,
    void* df,
    void* eforce,
    const void* velems,
    T kFactor)
{
    const int computeThreads = 64;
    int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeDForce_kernel<T, NNodes, Dim>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)rotations,
            (const T*)stiffness,
            (const T*)dx,
            (T*)eforce,
            kFactor);
    mycudaDebugError("ElementCorotationalFEMForceField_computeDForce_kernel");

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

#define INSTANTIATE_COROTATIONAL(T, NNodes) \
    template void ElementCorotationalFEMForceFieldCuda_addForce<T, NNodes, 3>( \
        unsigned int, unsigned int, unsigned int, const void*, const void*, \
        const void*, const void*, const void*, void*, void*, const void*); \
    template void ElementCorotationalFEMForceFieldCuda_addDForce<T, NNodes, 3>( \
        unsigned int, unsigned int, unsigned int, const void*, const void*, \
        const void*, const void*, void*, void*, const void*, T);

#define INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(T, NNodes) \
    template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<T, NNodes, 3>( \
        unsigned int, unsigned int, unsigned int, const void*, const void*, \
        const void*, const void*, const void*, void*, void*, void*, const void*);

INSTANTIATE_COROTATIONAL(float, 2)
INSTANTIATE_COROTATIONAL(float, 3)
INSTANTIATE_COROTATIONAL(float, 4)
INSTANTIATE_COROTATIONAL(float, 8)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(float, 3)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(float, 4)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(float, 8)

INSTANTIATE_COROTATIONAL(double, 2)
INSTANTIATE_COROTATIONAL(double, 3)
INSTANTIATE_COROTATIONAL(double, 4)
INSTANTIATE_COROTATIONAL(double, 8)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(double, 3)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(double, 4)
INSTANTIATE_COROTATIONAL_WITH_ROTATIONS(double, 8)

#undef INSTANTIATE_COROTATIONAL
#undef INSTANTIATE_COROTATIONAL_WITH_ROTATIONS

} // namespace sofa::gpu::cuda
