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
 * Templated on T for float/double support.
 */
template<typename T, int NNodes>
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
    T disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        disp[n * 3 + 0] = x[nodeId * 3 + 0] - x0[nodeId * 3 + 0];
        disp[n * 3 + 1] = x[nodeId * 3 + 1] - x0[nodeId * 3 + 1];
        disp[n * 3 + 2] = x[nodeId * 3 + 2] - x0[nodeId * 3 + 2];
    }

    // Symmetric block-matrix multiply: edf = K * disp
    const T* K = stiffness + elemId * NSymBlocks * 9;
    T edf[NNodes * 3];

    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        edf[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block
        {
            const T* Kii = K + diagIdx * 9;
            const T di0 = disp[ni * 3 + 0];
            const T di1 = disp[ni * 3 + 1];
            const T di2 = disp[ni * 3 + 2];
            edf[ni * 3 + 0] += Kii[0] * di0 + Kii[1] * di1 + Kii[2] * di2;
            edf[ni * 3 + 1] += Kii[3] * di0 + Kii[4] * di1 + Kii[5] * di2;
            edf[ni * 3 + 2] += Kii[6] * di0 + Kii[7] * di1 + Kii[8] * di2;
        }

        // Off-diagonal blocks
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * 9;

            {
                const T dj0 = disp[nj * 3 + 0];
                const T dj1 = disp[nj * 3 + 1];
                const T dj2 = disp[nj * 3 + 2];
                edf[ni * 3 + 0] += Kij[0] * dj0 + Kij[1] * dj1 + Kij[2] * dj2;
                edf[ni * 3 + 1] += Kij[3] * dj0 + Kij[4] * dj1 + Kij[5] * dj2;
                edf[ni * 3 + 2] += Kij[6] * dj0 + Kij[7] * dj1 + Kij[8] * dj2;
            }

            {
                const T di0 = disp[ni * 3 + 0];
                const T di1 = disp[ni * 3 + 1];
                const T di2 = disp[ni * 3 + 2];
                edf[nj * 3 + 0] += Kij[0] * di0 + Kij[3] * di1 + Kij[6] * di2;
                edf[nj * 3 + 1] += Kij[1] * di0 + Kij[4] * di1 + Kij[7] * di2;
                edf[nj * 3 + 2] += Kij[2] * di0 + Kij[5] * di1 + Kij[8] * di2;
            }
        }
    }

    // Write: eforce = -edf (minus sign from f -= K * displacement)
    T* out = eforce + elemId * NNodes * 3;
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
 */
template<typename T, int NNodes>
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
    T edx[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        edx[n * 3 + 0] = dx[nodeId * 3 + 0];
        edx[n * 3 + 1] = dx[nodeId * 3 + 1];
        edx[n * 3 + 2] = dx[nodeId * 3 + 2];
    }

    // Symmetric block-matrix multiply: edf = K * edx
    const T* K = stiffness + elemId * NSymBlocks * 9;
    T edf[NNodes * 3];

    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        edf[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        {
            const T* Kii = K + diagIdx * 9;
            const T di0 = edx[ni * 3 + 0];
            const T di1 = edx[ni * 3 + 1];
            const T di2 = edx[ni * 3 + 2];
            edf[ni * 3 + 0] += Kii[0] * di0 + Kii[1] * di1 + Kii[2] * di2;
            edf[ni * 3 + 1] += Kii[3] * di0 + Kii[4] * di1 + Kii[5] * di2;
            edf[ni * 3 + 2] += Kii[6] * di0 + Kii[7] * di1 + Kii[8] * di2;
        }

        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * 9;

            {
                const T dj0 = edx[nj * 3 + 0];
                const T dj1 = edx[nj * 3 + 1];
                const T dj2 = edx[nj * 3 + 2];
                edf[ni * 3 + 0] += Kij[0] * dj0 + Kij[1] * dj1 + Kij[2] * dj2;
                edf[ni * 3 + 1] += Kij[3] * dj0 + Kij[4] * dj1 + Kij[5] * dj2;
                edf[ni * 3 + 2] += Kij[6] * dj0 + Kij[7] * dj1 + Kij[8] * dj2;
            }

            {
                const T di0 = edx[ni * 3 + 0];
                const T di1 = edx[ni * 3 + 1];
                const T di2 = edx[ni * 3 + 2];
                edf[nj * 3 + 0] += Kij[0] * di0 + Kij[3] * di1 + Kij[6] * di2;
                edf[nj * 3 + 1] += Kij[1] * di0 + Kij[4] * di1 + Kij[7] * di2;
                edf[nj * 3 + 2] += Kij[2] * di0 + Kij[5] * di1 + Kij[8] * di2;
            }
        }
    }

    // Write: eforce = -kFactor * edf
    T* out = eforce + elemId * NNodes * 3;
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
 */
template<typename T>
__global__ void ElementLinearSmallStrainFEMForceField_gatherForce_kernel(
    int nbVertex,
    int maxElemPerVertex,
    const int* __restrict__ velems,
    const T* __restrict__ eforce,
    T* df)
{
    const int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId >= nbVertex) return;

    T fx = T(0), fy = T(0), fz = T(0);

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

template<typename T>
static void launchGather(
    unsigned int nbVertex,
    unsigned int maxElemPerVertex,
    const void* velems,
    const void* eforce,
    void* f)
{
    const int gatherThreads = 256;
    const int numBlocks = (nbVertex + gatherThreads - 1) / gatherThreads;
    ElementLinearSmallStrainFEMForceField_gatherForce_kernel<T>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_gatherForce_kernel");
}

template<typename T, int NNodes>
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
    ElementLinearSmallStrainFEMForceField_computeForce_kernel<T, NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)stiffness,
            (const T*)x,
            (const T*)x0,
            (T*)eforce);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_computeForce_kernel");

    launchGather<T>(nbVertex, maxElemPerVertex, velems, eforce, f);
}

template<typename T, int NNodes>
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
    T kFactor)
{
    const int computeThreads = 64;
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementLinearSmallStrainFEMForceField_computeDForce_kernel<T, NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)stiffness,
            (const T*)dx,
            (T*)eforce,
            kFactor);
    mycudaDebugError("ElementLinearSmallStrainFEMForceField_computeDForce_kernel");

    launchGather<T>(nbVertex, maxElemPerVertex, velems, eforce, df);
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
        case 2: launchAddForce<float, 2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<float, 3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<float, 4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<float, 8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
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
        case 2: launchAddDForce<float, 2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<float, 3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<float, 4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<float, 8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

void ElementLinearSmallStrainFEMForceFieldCuda3d_addForce(
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
        case 2: launchAddForce<double, 2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<double, 3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<double, 4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<double, 8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, x, x0, f, eforce, velems); break;
    }
}

void ElementLinearSmallStrainFEMForceFieldCuda3d_addDForce(
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
    double kFactor)
{
    switch (nbNodesPerElem)
    {
        case 2: launchAddDForce<double, 2>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<double, 3>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<double, 4>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<double, 8>(nbElem, nbVertex, maxElemPerVertex, elements, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
