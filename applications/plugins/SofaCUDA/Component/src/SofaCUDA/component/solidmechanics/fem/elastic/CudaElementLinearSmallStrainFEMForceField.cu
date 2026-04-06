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

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

/**
 * Kernel for addForce: Compute per-element force from displacement (1 thread per element).
 *
 * f = -K * (x - x0)
 * Templated on NNodes (compile-time) for full loop unrolling.
 * Hardcoded Dim=3 (CudaVec3f only).
 */
template<int NNodes>
__global__ void ElementLinearSmallStrainFEMForceFieldCuda3f_computeForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const float* __restrict__ stiffness,
    const float* __restrict__ x,
    const float* __restrict__ x0,
    float* __restrict__ eforce)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather displacement = x - x0 for this element's nodes
    float disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        disp[n * 3 + 0] = x[nodeId * 3 + 0] - x0[nodeId * 3 + 0];
        disp[n * 3 + 1] = x[nodeId * 3 + 1] - x0[nodeId * 3 + 1];
        disp[n * 3 + 2] = x[nodeId * 3 + 2] - x0[nodeId * 3 + 2];
    }

    // Symmetric block-matrix multiply: edf = K * disp
    const float* K = stiffness + elemId * NSymBlocks * 9;
    float edf[NNodes * 3];

    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        edf[i] = 0.0f;

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block
        {
            const float* Kii = K + diagIdx * 9;
            const float di0 = disp[ni * 3 + 0];
            const float di1 = disp[ni * 3 + 1];
            const float di2 = disp[ni * 3 + 2];
            edf[ni * 3 + 0] += Kii[0] * di0 + Kii[1] * di1 + Kii[2] * di2;
            edf[ni * 3 + 1] += Kii[3] * di0 + Kii[4] * di1 + Kii[5] * di2;
            edf[ni * 3 + 2] += Kii[6] * di0 + Kii[7] * di1 + Kii[8] * di2;
        }

        // Off-diagonal blocks
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const float* Kij = K + symIdx * 9;

            {
                const float dj0 = disp[nj * 3 + 0];
                const float dj1 = disp[nj * 3 + 1];
                const float dj2 = disp[nj * 3 + 2];
                edf[ni * 3 + 0] += Kij[0] * dj0 + Kij[1] * dj1 + Kij[2] * dj2;
                edf[ni * 3 + 1] += Kij[3] * dj0 + Kij[4] * dj1 + Kij[5] * dj2;
                edf[ni * 3 + 2] += Kij[6] * dj0 + Kij[7] * dj1 + Kij[8] * dj2;
            }

            {
                const float di0 = disp[ni * 3 + 0];
                const float di1 = disp[ni * 3 + 1];
                const float di2 = disp[ni * 3 + 2];
                edf[nj * 3 + 0] += Kij[0] * di0 + Kij[3] * di1 + Kij[6] * di2;
                edf[nj * 3 + 1] += Kij[1] * di0 + Kij[4] * di1 + Kij[7] * di2;
                edf[nj * 3 + 2] += Kij[2] * di0 + Kij[5] * di1 + Kij[8] * di2;
            }
        }
    }

    // Write: eforce = -edf (minus sign from f -= K * displacement)
    float* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        out[n * 3 + 0] = -edf[n * 3 + 0];
        out[n * 3 + 1] = -edf[n * 3 + 1];
        out[n * 3 + 2] = -edf[n * 3 + 2];
    }
}

/**
 * Kernel for addDForce: Compute per-element dForce (1 thread per element).
 *
 * df = -kFactor * K * dx
 * Templated on NNodes (compile-time) for full loop unrolling.
 * Hardcoded Dim=3 (CudaVec3f only).
 */
template<int NNodes>
__global__ void ElementLinearSmallStrainFEMForceFieldCuda3f_computeDForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const float* __restrict__ stiffness,
    const float* __restrict__ dx,
    float* __restrict__ eforce,
    float kFactor)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather dx for this element's nodes
    float edx[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        edx[n * 3 + 0] = dx[nodeId * 3 + 0];
        edx[n * 3 + 1] = dx[nodeId * 3 + 1];
        edx[n * 3 + 2] = dx[nodeId * 3 + 2];
    }

    // Symmetric block-matrix multiply: edf = K * edx
    const float* K = stiffness + elemId * NSymBlocks * 9;
    float edf[NNodes * 3];

    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        edf[i] = 0.0f;

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block (ni, ni): Kii * edx[ni]
        {
            const float* Kii = K + diagIdx * 9;
            const float di0 = edx[ni * 3 + 0];
            const float di1 = edx[ni * 3 + 1];
            const float di2 = edx[ni * 3 + 2];
            edf[ni * 3 + 0] += Kii[0] * di0 + Kii[1] * di1 + Kii[2] * di2;
            edf[ni * 3 + 1] += Kii[3] * di0 + Kii[4] * di1 + Kii[5] * di2;
            edf[ni * 3 + 2] += Kii[6] * di0 + Kii[7] * di1 + Kii[8] * di2;
        }

        // Off-diagonal blocks (ni, nj) for nj > ni
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const float* Kij = K + symIdx * 9;

            // Forward: edf[ni] += Kij * edx[nj]
            {
                const float dj0 = edx[nj * 3 + 0];
                const float dj1 = edx[nj * 3 + 1];
                const float dj2 = edx[nj * 3 + 2];
                edf[ni * 3 + 0] += Kij[0] * dj0 + Kij[1] * dj1 + Kij[2] * dj2;
                edf[ni * 3 + 1] += Kij[3] * dj0 + Kij[4] * dj1 + Kij[5] * dj2;
                edf[ni * 3 + 2] += Kij[6] * dj0 + Kij[7] * dj1 + Kij[8] * dj2;
            }

            // Symmetric: edf[nj] += Kij^T * edx[ni]
            {
                const float di0 = edx[ni * 3 + 0];
                const float di1 = edx[ni * 3 + 1];
                const float di2 = edx[ni * 3 + 2];
                edf[nj * 3 + 0] += Kij[0] * di0 + Kij[3] * di1 + Kij[6] * di2;
                edf[nj * 3 + 1] += Kij[1] * di0 + Kij[4] * di1 + Kij[7] * di2;
                edf[nj * 3 + 2] += Kij[2] * di0 + Kij[5] * di1 + Kij[8] * di2;
            }
        }
    }

    // Write: eforce = -kFactor * edf
    float* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        out[n * 3 + 0] = -kFactor * edf[n * 3 + 0];
        out[n * 3 + 1] = -kFactor * edf[n * 3 + 1];
        out[n * 3 + 2] = -kFactor * edf[n * 3 + 2];
    }
}

/**
 * Gather per-vertex forces (1 thread per vertex).
 *
 * Shared by both addForce and addDForce.
 * No atomics: each vertex handled by exactly one thread.
 * velems is SoA: velems[s * nbVertex + vertexId], 0-terminated.
 * Each entry is (elemId * NNodes + localNode + 1), with 0 as sentinel.
 */
__global__ void ElementLinearSmallStrainFEMForceFieldCuda3f_gatherForce_kernel(
    int nbVertex,
    int maxElemPerVertex,
    const int* __restrict__ velems,
    const float* __restrict__ eforce,
    float* df)
{
    const int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId >= nbVertex) return;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int s = 0; s < maxElemPerVertex; ++s)
    {
        const int idx = velems[s * nbVertex + vertexId];
        if (idx == 0) break;
        const int base = (idx - 1) * 3;
        fx += eforce[base + 0];
        fy += eforce[base + 1];
        fz += eforce[base + 2];
    }

    df[vertexId * 3 + 0] += fx;
    df[vertexId * 3 + 1] += fy;
    df[vertexId * 3 + 2] += fz;
}

static void launchGather(
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* velems,
    const void* eforce,
    void* f)
{
    const int gatherThreads = 256;
    const int numBlocks = (nbVertex + gatherThreads - 1) / gatherThreads;
    ElementLinearSmallStrainFEMForceFieldCuda3f_gatherForce_kernel
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const float*)eforce,
            (float*)f);
    mycudaDebugError("ElementLinearSmallStrainFEMForceFieldCuda3f_gatherForce_kernel");
}

template<int NNodes>
static void launchAddForce(
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
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementLinearSmallStrainFEMForceFieldCuda3f_computeForce_kernel<NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const float*)stiffness,
            (const float*)x,
            (const float*)x0,
            (float*)eforce);
    mycudaDebugError("ElementLinearSmallStrainFEMForceFieldCuda3f_computeForce_kernel");

    launchGather(nbVertex, maxElemPerVertex, velems, eforce, f);
}

template<int NNodes>
static void launchAddDForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* stiffness,
    const void* dx,
    void* df,
    void* eforce,
    const void* velems,
    float kFactor)
{
    const int computeThreads = 64;
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementLinearSmallStrainFEMForceFieldCuda3f_computeDForce_kernel<NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const float*)stiffness,
            (const float*)dx,
            (float*)eforce,
            kFactor);
    mycudaDebugError("ElementLinearSmallStrainFEMForceFieldCuda3f_computeDForce_kernel");

    launchGather(nbVertex, maxElemPerVertex, velems, eforce, df);
}

extern "C"
{

void ElementLinearSmallStrainFEMForceFieldCuda3f_addForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* stiffness,
    const void* x,
    const void* x0,
    void* f,
    void* eforce,
    const void* velems)
{
    switch (nbNodesPerElem)
    {
        case 2: launchAddForce<2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
    }
}

void ElementLinearSmallStrainFEMForceFieldCuda3f_addDForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* stiffness,
    const void* dx,
    void* df,
    void* eforce,
    const void* velems,
    float kFactor)
{
    switch (nbNodesPerElem)
    {
        case 2: launchAddDForce<2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
