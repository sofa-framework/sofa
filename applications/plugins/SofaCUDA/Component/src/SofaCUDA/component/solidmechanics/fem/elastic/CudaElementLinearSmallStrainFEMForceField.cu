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
#include <cuda.h>

namespace sofa
{
namespace gpu
{
namespace cuda
{

/**
 * Kernel for addForce: Compute per-element force from displacement (1 thread per element).
 *
 * f = -K * (x - x0)
 * Templated on NNodes and Dim (compile-time) for full loop unrolling.
 * Templated on T for float/double support.
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

    // Gather displacement = x - x0 for this element's nodes
    T disp[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            disp[n * Dim + d] = x[nodeId * Dim + d] - x0[nodeId * Dim + d];
    }

    // Symmetric block-matrix multiply: edf = K * disp
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];

    #pragma unroll
    for (int i = 0; i < NNodes * Dim; ++i)
        edf[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block
        {
            const T* Kii = K + diagIdx * Dim * Dim;
            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kii[di * Dim + dj] * disp[ni * Dim + dj];
                edf[ni * Dim + di] += sum;
            }
        }

        // Off-diagonal blocks
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * Dim * Dim;

            // Kij * disp_j -> edf_i
            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kij[di * Dim + dj] * disp[nj * Dim + dj];
                edf[ni * Dim + di] += sum;
            }

            // Kij^T * disp_i -> edf_j
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
            {
                T sum = T(0);
                #pragma unroll
                for (int di = 0; di < Dim; ++di)
                    sum += Kij[di * Dim + dj] * disp[ni * Dim + di];
                edf[nj * Dim + dj] += sum;
            }
        }
    }

    // Write: eforce = -edf (minus sign from f -= K * displacement)
    T* out = eforce + elemId * NNodes * Dim;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            out[n * Dim + d] = -edf[n * Dim + d];
    }
}

/**
 * Kernel for addDForce: Compute per-element dForce (1 thread per element).
 *
 * df = -kFactor * K * dx
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

    // Gather dx for this element's nodes
    T edx[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            edx[n * Dim + d] = dx[nodeId * Dim + d];
    }

    // Symmetric block-matrix multiply: edf = K * edx
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];

    #pragma unroll
    for (int i = 0; i < NNodes * Dim; ++i)
        edf[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        {
            const T* Kii = K + diagIdx * Dim * Dim;
            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kii[di * Dim + dj] * edx[ni * Dim + dj];
                edf[ni * Dim + di] += sum;
            }
        }

        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * Dim * Dim;

            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kij[di * Dim + dj] * edx[nj * Dim + dj];
                edf[ni * Dim + di] += sum;
            }

            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
            {
                T sum = T(0);
                #pragma unroll
                for (int di = 0; di < Dim; ++di)
                    sum += Kij[di * Dim + dj] * edx[ni * Dim + di];
                edf[nj * Dim + dj] += sum;
            }
        }
    }

    // Write: eforce = -kFactor * edf
    T* out = eforce + elemId * NNodes * Dim;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            out[n * Dim + d] = -kFactor * edf[n * Dim + d];
    }
}

/**
 * Gather per-vertex forces (1 thread per vertex).
 */
template<typename T, int Dim>
__global__ void ElementLinearSmallStrainFEMForceField_gatherForce_kernel(
    int nbVertex,
    int maxElemPerVertex,
    const int* __restrict__ velems,
    const T* __restrict__ eforce,
    T* df)
{
    const int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId >= nbVertex) return;

    T acc[Dim];
    #pragma unroll
    for (int d = 0; d < Dim; ++d)
        acc[d] = T(0);

    for (int s = 0; s < maxElemPerVertex; ++s)
    {
        const int idx = velems[s * nbVertex + vertexId];
        if (idx == 0) break;
        const int base = (idx - 1) * Dim;
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            acc[d] += eforce[base + d];
    }

    #pragma unroll
    for (int d = 0; d < Dim; ++d)
        df[vertexId * Dim + d] += acc[d];
}

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
    ElementLinearSmallStrainFEMForceField_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_gatherForce_kernel");
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
    ElementLinearSmallStrainFEMForceField_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)df);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_gatherForce_kernel");
}

// Explicit template instantiations for all supported (T, NNodes, Dim) combinations
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<float, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<float, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<float, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<float, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<double, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<double, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<double, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementLinearSmallStrainFEMForceFieldCuda_addForce<double, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*);

template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<float, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<float, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<float, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<float, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<double, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<double, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<double, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementLinearSmallStrainFEMForceFieldCuda_addDForce<double, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, void*, void*, const void*, double);

} // namespace cuda
} // namespace gpu
} // namespace sofa
