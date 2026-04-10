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
#pragma once

#include <cuda.h>

namespace sofa::gpu::cuda
{

//=============================================================================
// Math utilities
//=============================================================================

template<typename T>
__device__ inline T myRsqrt(T x);

template<>
__device__ inline float myRsqrt<float>(float x) { return rsqrtf(x); }

template<>
__device__ inline double myRsqrt<double>(double x) { return rsqrt(x); }

//=============================================================================
// 3x3 Matrix operations (row-major)
//=============================================================================

template<typename T>
__device__ inline void mat3Mul(const T* A, const T* B, T* C)
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

template<typename T>
__device__ inline void mat3MulTranspose(const T* A, const T* BT, T* C)
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

template<typename T>
__device__ inline void mat3TransposeMul(const T* A, const T* B, T* C)
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

//=============================================================================
// Rotation frame computation
//=============================================================================

template<typename T>
__device__ inline void computeTriangleFrame(const T* ex, T* frame)
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

template<typename T>
__device__ inline void computeHexahedronFrame(const T* ex, T* frame)
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

//=============================================================================
// Symmetric block-matrix multiply
//=============================================================================

template<typename T, int NNodes, int Dim>
__device__ inline void symBlockMatMul(const T* K, const T* in, T* out)
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

            #pragma unroll
            for (int di = 0; di < Dim; ++di)
            {
                T sum = T(0);
                #pragma unroll
                for (int dj = 0; dj < Dim; ++dj)
                    sum += Kij[di * Dim + dj] * in[nj * Dim + dj];
                out[ni * Dim + di] += sum;
            }

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

//=============================================================================
// Gather kernel
//=============================================================================

template<typename T, int Dim>
__global__ void ElementFEM_gatherForce_kernel(
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

} // namespace sofa::gpu::cuda
