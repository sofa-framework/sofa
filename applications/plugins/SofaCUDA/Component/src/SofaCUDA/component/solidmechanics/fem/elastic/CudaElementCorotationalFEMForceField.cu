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
 * Kernel 1: Compute per-element dForce (1 thread per element).
 *
 * Templated on NNodes (compile-time) for full loop unrolling.
 * Hardcoded Dim=3 (CudaVec3f only).
 *
 * Connectivity is SoA: elements[nodeIdx * nbElem + elemId].
 * Stiffness is in block format: K[(ni * NNodes + nj) * 9 + di * 3 + dj].
 */
template<int NNodes>
__global__ void ElementCorotationalFEMForceFieldCuda3f_computeDForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const float* __restrict__ rotations,
    const float* __restrict__ stiffness,
    const float* __restrict__ dx,
    float* __restrict__ eforce,
    float kFactor)
{
    constexpr int NDofs = NNodes * 3;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load rotation matrix R (3x3, row-major)
    const float* Rptr = rotations + elemId * 9;
    float R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        R[i] = Rptr[i];

    // Gather dx and rotate into reference frame: rdx[n] = R^T * dx[node[n]]
    float rdx[NDofs];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        const float dx_x = dx[nodeId * 3 + 0];
        const float dx_y = dx[nodeId * 3 + 1];
        const float dx_z = dx[nodeId * 3 + 2];

        rdx[n * 3 + 0] = R[0] * dx_x + R[3] * dx_y + R[6] * dx_z;
        rdx[n * 3 + 1] = R[1] * dx_x + R[4] * dx_y + R[7] * dx_z;
        rdx[n * 3 + 2] = R[2] * dx_x + R[5] * dx_y + R[8] * dx_z;
    }

    // Block-matrix multiply: edf = K * rdx
    const float* K = stiffness + elemId * NNodes * NNodes * 9;
    float edf[NDofs];
    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        float fi0 = 0.0f, fi1 = 0.0f, fi2 = 0.0f;
        #pragma unroll
        for (int nj = 0; nj < NNodes; ++nj)
        {
            const float* Kij = K + (ni * NNodes + nj) * 9;
            const float rj0 = rdx[nj * 3 + 0];
            const float rj1 = rdx[nj * 3 + 1];
            const float rj2 = rdx[nj * 3 + 2];
            fi0 += Kij[0] * rj0 + Kij[1] * rj1 + Kij[2] * rj2;
            fi1 += Kij[3] * rj0 + Kij[4] * rj1 + Kij[5] * rj2;
            fi2 += Kij[6] * rj0 + Kij[7] * rj1 + Kij[8] * rj2;
        }
        edf[ni * 3 + 0] = fi0;
        edf[ni * 3 + 1] = fi1;
        edf[ni * 3 + 2] = fi2;
    }

    // Rotate back and write: eforce = -kFactor * R * edf
    float* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const float e0 = edf[n * 3 + 0];
        const float e1 = edf[n * 3 + 1];
        const float e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -kFactor * (R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -kFactor * (R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -kFactor * (R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Kernel 2: Gather per-vertex forces (1 thread per vertex).
 *
 * No atomics: each vertex handled by exactly one thread.
 * velems is SoA: velems[s * nbVertex + vertexId], 0-terminated.
 * Each entry is (elemId * NNodes + localNode + 1), with 0 as sentinel.
 */
__global__ void ElementCorotationalFEMForceFieldCuda3f_gatherDForce_kernel(
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

template<int NNodes>
static void launchAddDForce(
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
    float kFactor)
{
    const int computeThreads = 64;
    const int gatherThreads = 256;

    {
        const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
        ElementCorotationalFEMForceFieldCuda3f_computeDForce_kernel<NNodes>
            <<<numBlocks, computeThreads>>>(
                nbElem,
                (const int*)elements,
                (const float*)rotations,
                (const float*)stiffness,
                (const float*)dx,
                (float*)eforce,
                kFactor);
        mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_computeDForce_kernel");
    }

    {
        const int numBlocks = (nbVertex + gatherThreads - 1) / gatherThreads;
        ElementCorotationalFEMForceFieldCuda3f_gatherDForce_kernel
            <<<numBlocks, gatherThreads>>>(
                nbVertex,
                maxElemPerVertex,
                (const int*)velems,
                (const float*)eforce,
                (float*)df);
        mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_gatherDForce_kernel");
    }
}

extern "C"
{

void ElementCorotationalFEMForceFieldCuda3f_addDForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
    unsigned int maxElemPerVertex,
    const void* elements,
    const void* rotations,
    const void* stiffness,
    const void* dx,
    void* df,
    void* eforce,
    const void* velems,
    float kFactor)
{
    switch (nbNodesPerElem)
    {
        case 2: launchAddDForce<2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
