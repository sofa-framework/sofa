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

template<typename T>
__device__ T myRsqrt(T x);
template<> __device__ float myRsqrt<float>(float x) { return rsqrtf(x); }
template<> __device__ double myRsqrt<double>(double x) { return rsqrt(x); }

/**
 * Device helper: 3x3 matrix multiply C = A * B (row-major)
 */
template<typename T>
__device__ void mat3Mul(const T* A, const T* B, T* C)
{
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j]
                          + A[i * 3 + 1] * B[1 * 3 + j]
                          + A[i * 3 + 2] * B[2 * 3 + j];
        }
    }
}

/**
 * Device helper: C = A * B^T (row-major)
 */
template<typename T>
__device__ void mat3MulTranspose(const T* A, const T* BT, T* C)
{
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            C[i * 3 + j] = A[i * 3 + 0] * BT[j * 3 + 0]
                          + A[i * 3 + 1] * BT[j * 3 + 1]
                          + A[i * 3 + 2] * BT[j * 3 + 2];
        }
    }
}

/**
 * Device helper: C = A^T * B (row-major)
 * Matches SOFA's Mat::multTranspose(B) which computes this^T * B.
 */
template<typename T>
__device__ void mat3TransposeMul(const T* A, const T* B, T* C)
{
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            C[i * 3 + j] = A[0 * 3 + i] * B[0 * 3 + j]
                          + A[1 * 3 + i] * B[1 * 3 + j]
                          + A[2 * 3 + i] * B[2 * 3 + j];
        }
    }
}

/**
 * Device helper: compute rotation frame from first 3 nodes (TriangleRotation).
 */
template<typename T>
__device__ void computeTriangleFrame(const T* ex, T* frame)
{
    T ax = ex[3] - ex[0], ay = ex[4] - ex[1], az = ex[5] - ex[2];
    T invLen = myRsqrt(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    T bx = ex[6] - ex[0], by = ex[7] - ex[1], bz = ex[8] - ex[2];

    T cx = ay * bz - az * by;
    T cy = az * bx - ax * bz;
    T cz = ax * by - ay * bx;
    invLen = myRsqrt(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

/**
 * Device helper: compute rotation frame from 8 hexahedron nodes (HexahedronRotation).
 */
template<typename T>
__device__ void computeHexahedronFrame(const T* ex, T* frame)
{
    const T quarter = T(0.25);

    T ax = ((ex[1*3+0] - ex[0*3+0]) + (ex[2*3+0] - ex[3*3+0])
          + (ex[5*3+0] - ex[4*3+0]) + (ex[6*3+0] - ex[7*3+0])) * quarter;
    T ay = ((ex[1*3+1] - ex[0*3+1]) + (ex[2*3+1] - ex[3*3+1])
          + (ex[5*3+1] - ex[4*3+1]) + (ex[6*3+1] - ex[7*3+1])) * quarter;
    T az = ((ex[1*3+2] - ex[0*3+2]) + (ex[2*3+2] - ex[3*3+2])
          + (ex[5*3+2] - ex[4*3+2]) + (ex[6*3+2] - ex[7*3+2])) * quarter;

    T bx = ((ex[3*3+0] - ex[0*3+0]) + (ex[2*3+0] - ex[1*3+0])
          + (ex[7*3+0] - ex[4*3+0]) + (ex[6*3+0] - ex[5*3+0])) * quarter;
    T by = ((ex[3*3+1] - ex[0*3+1]) + (ex[2*3+1] - ex[1*3+1])
          + (ex[7*3+1] - ex[4*3+1]) + (ex[6*3+1] - ex[5*3+1])) * quarter;
    T bz = ((ex[3*3+2] - ex[0*3+2]) + (ex[2*3+2] - ex[1*3+2])
          + (ex[7*3+2] - ex[4*3+2]) + (ex[6*3+2] - ex[5*3+2])) * quarter;

    T invLen = myRsqrt(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    T cx = ay * bz - az * by;
    T cy = az * bx - ax * bz;
    T cz = ax * by - ay * bx;
    invLen = myRsqrt(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

/**
 * Symmetric block-matrix multiply: out = K * in
 * Templated on Dim for generic spatial dimensions.
 */
template<typename T, int NNodes, int Dim>
__device__ void symBlockMatMul(const T* K, const T* in, T* out)
{
    #pragma unroll
    for (int i = 0; i < NNodes * Dim; ++i)
        out[i] = T(0);

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
                    sum += Kii[di * Dim + dj] * in[ni * Dim + dj];
                out[ni * Dim + di] += sum;
            }
        }

        // Off-diagonal blocks
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * Dim * Dim;

            // Kij * in_j -> out_i
            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kij[di * Dim + dj] * in[nj * Dim + dj];
                out[ni * Dim + di] += sum;
            }

            // Kij^T * in_i -> out_j
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
            {
                T sum = T(0);
                #pragma unroll
                for (int di = 0; di < Dim; ++di)
                    sum += Kij[di * Dim + dj] * in[ni * Dim + di];
                out[nj * Dim + dj] += sum;
            }
        }
    }
}

/**
 * Combined kernel: compute rotations AND per-element forces in one pass.
 * Rotation computation is inherently 3D (cross products).
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
    const T invN = T(1) / T(NNodes);

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    T ex[NNodes * Dim], ex0[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
        {
            ex[n * Dim + d] = x[nodeId * Dim + d];
            ex0[n * Dim + d] = x0[nodeId * Dim + d];
        }
    }

    T frame[Dim * Dim];
    if constexpr (NNodes == 8)
        computeHexahedronFrame(ex, frame);
    else
        computeTriangleFrame(ex, frame);

    // R = frame^T * initRot
    const T* irt = initRotTransposed + elemId * Dim * Dim;
    T R[Dim * Dim];
    mat3TransposeMul(frame, irt, R);

    T* Rout = rotationsOut + elemId * Dim * Dim;
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        Rout[i] = R[i];

    T center[Dim], center0[Dim];
    #pragma unroll
    for (int d = 0; d < Dim; ++d)
    {
        center[d] = T(0);
        center0[d] = T(0);
    }
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
        {
            center[d] += ex[n * Dim + d];
            center0[d] += ex0[n * Dim + d];
        }
    }
    #pragma unroll
    for (int d = 0; d < Dim; ++d)
    {
        center[d] *= invN;
        center0[d] *= invN;
    }

    T disp[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        // R^T * (x_n - center)
        T diff[Dim];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            diff[d] = ex[n * Dim + d] - center[d];

        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T rotated = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                rotated += R[dj * Dim + di] * diff[dj];
            disp[n * Dim + di] = rotated - (ex0[n * Dim + di] - center0[di]);
        }
    }

    T edf[NNodes * Dim];
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    symBlockMatMul<T, NNodes, Dim>(K, disp, edf);

    T* out = eforce + elemId * NNodes * Dim;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        // R * edf_n, negated
        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T sum = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                sum += R[di * Dim + dj] * edf[n * Dim + dj];
            out[n * Dim + di] = -sum;
        }
    }
}

/**
 * Kernel for addForce: Compute per-element force (1 thread per element).
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
    const T invN = T(1) / T(NNodes);

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    const T* Rptr = rotations + elemId * Dim * Dim;
    T R[Dim * Dim];
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        R[i] = Rptr[i];

    T ex[NNodes * Dim], ex0[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
        {
            ex[n * Dim + d] = x[nodeId * Dim + d];
            ex0[n * Dim + d] = x0[nodeId * Dim + d];
        }
    }

    T center[Dim], center0[Dim];
    #pragma unroll
    for (int d = 0; d < Dim; ++d)
    {
        center[d] = T(0);
        center0[d] = T(0);
    }
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
        {
            center[d] += ex[n * Dim + d];
            center0[d] += ex0[n * Dim + d];
        }
    }
    #pragma unroll
    for (int d = 0; d < Dim; ++d)
    {
        center[d] *= invN;
        center0[d] *= invN;
    }

    T disp[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        T diff[Dim];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            diff[d] = ex[n * Dim + d] - center[d];

        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T rotated = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                rotated += R[dj * Dim + di] * diff[dj];
            disp[n * Dim + di] = rotated - (ex0[n * Dim + di] - center0[di]);
        }
    }

    T edf[NNodes * Dim];
    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    symBlockMatMul<T, NNodes, Dim>(K, disp, edf);

    T* out = eforce + elemId * NNodes * Dim;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T sum = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                sum += R[di * Dim + dj] * edf[n * Dim + dj];
            out[n * Dim + di] = -sum;
        }
    }
}

/**
 * Kernel for addDForce: Compute per-element dForce (1 thread per element).
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

    const T* Rptr = rotations + elemId * Dim * Dim;
    T R[Dim * Dim];
    #pragma unroll
    for (int i = 0; i < Dim * Dim; ++i)
        R[i] = Rptr[i];

    // R^T * dx for each node
    T rdx[NNodes * Dim];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        T nodeDx[Dim];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            nodeDx[d] = dx[nodeId * Dim + d];

        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T sum = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                sum += R[dj * Dim + di] * nodeDx[dj];
            rdx[n * Dim + di] = sum;
        }
    }

    const T* K = stiffness + elemId * NSymBlocks * Dim * Dim;
    T edf[NNodes * Dim];
    symBlockMatMul<T, NNodes, Dim>(K, rdx, edf);

    // R * edf, scaled by -kFactor
    T* out = eforce + elemId * NNodes * Dim;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T sum = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                sum += R[di * Dim + dj] * edf[n * Dim + dj];
            out[n * Dim + di] = -kFactor * sum;
        }
    }
}

/**
 * Gather per-vertex forces (1 thread per vertex).
 */
template<typename T, int Dim>
__global__ void ElementCorotationalFEMForceField_gatherForce_kernel(
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

// ===================== Launch functions (C++ templates) =====================

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
    ElementCorotationalFEMForceField_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementCorotationalFEMForceField_gatherForce_kernel");
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
    ElementCorotationalFEMForceField_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementCorotationalFEMForceField_gatherForce_kernel");
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
    ElementCorotationalFEMForceField_gatherForce_kernel<T, Dim>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)df);
    mycudaDebugError("ElementCorotationalFEMForceField_gatherForce_kernel");
}

// ===================== Explicit template instantiations =====================

// addForceWithRotations: only NNodes >= 3 (triangle/quad/hex rotation methods)
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<float, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<float, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<float, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<double, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<double, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForceWithRotations<double, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, void*, const void*);

// addForce: all element types
template void ElementCorotationalFEMForceFieldCuda_addForce<float, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<float, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<float, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<float, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<double, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<double, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<double, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);
template void ElementCorotationalFEMForceFieldCuda_addForce<double, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*);

// addDForce: all element types
template void ElementCorotationalFEMForceFieldCuda_addDForce<float, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementCorotationalFEMForceFieldCuda_addDForce<float, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementCorotationalFEMForceFieldCuda_addDForce<float, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementCorotationalFEMForceFieldCuda_addDForce<float, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, float);
template void ElementCorotationalFEMForceFieldCuda_addDForce<double, 2, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementCorotationalFEMForceFieldCuda_addDForce<double, 3, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementCorotationalFEMForceFieldCuda_addDForce<double, 4, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, double);
template void ElementCorotationalFEMForceFieldCuda_addDForce<double, 8, 3>(unsigned int, unsigned int, unsigned int, const void*, const void*, const void*, const void*, void*, void*, const void*, double);

} // namespace cuda
} // namespace gpu
} // namespace sofa
