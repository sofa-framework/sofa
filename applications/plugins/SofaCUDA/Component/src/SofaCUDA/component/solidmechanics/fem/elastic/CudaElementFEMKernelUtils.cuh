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
// 3x3 Matrix operations (row-major storage)
//=============================================================================

/// C = A * B
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

/// C = A * B^T
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

/// C = A^T * B
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

/// out = R * in (rotate a 3D vector)
template<typename T>
__device__ inline void rotateVector(const T* R, const T* in, T* out)
{
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        out[i] = R[i * 3 + 0] * in[0]
               + R[i * 3 + 1] * in[1]
               + R[i * 3 + 2] * in[2];
    }
}

/// out = R^T * in (rotate a 3D vector by transpose)
template<typename T>
__device__ inline void rotateVectorTranspose(const T* R, const T* in, T* out)
{
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        out[i] = R[0 * 3 + i] * in[0]
               + R[1 * 3 + i] * in[1]
               + R[2 * 3 + i] * in[2];
    }
}

//=============================================================================
// Rotation frame computation
//=============================================================================

/// Compute rotation frame from first 3 nodes (for Triangle, Quad, Tetrahedron)
template<typename T>
__device__ inline void computeTriangleFrame(const T* pos, T* frame)
{
    // X axis: normalized (p1 - p0)
    T ax = pos[3] - pos[0], ay = pos[4] - pos[1], az = pos[5] - pos[2];
    T invLen = myRsqrt(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    // Temp vector b = p2 - p0
    T bx = pos[6] - pos[0], by = pos[7] - pos[1], bz = pos[8] - pos[2];

    // Z axis: normalized cross(a, b)
    T cx = ay * bz - az * by;
    T cy = az * bx - ax * bz;
    T cz = ax * by - ay * bx;
    invLen = myRsqrt(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    // Y axis: cross(z, x)
    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    // Store row-major: frame[row][col] = frame[row * 3 + col]
    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

/// Compute rotation frame from 8 hexahedron nodes
template<typename T>
__device__ inline void computeHexahedronFrame(const T* pos, T* frame)
{
    const T quarter = T(0.25);

    // Average X direction from 4 edge pairs
    T ax = ((pos[1*3+0] - pos[0*3+0]) + (pos[2*3+0] - pos[3*3+0])
          + (pos[5*3+0] - pos[4*3+0]) + (pos[6*3+0] - pos[7*3+0])) * quarter;
    T ay = ((pos[1*3+1] - pos[0*3+1]) + (pos[2*3+1] - pos[3*3+1])
          + (pos[5*3+1] - pos[4*3+1]) + (pos[6*3+1] - pos[7*3+1])) * quarter;
    T az = ((pos[1*3+2] - pos[0*3+2]) + (pos[2*3+2] - pos[3*3+2])
          + (pos[5*3+2] - pos[4*3+2]) + (pos[6*3+2] - pos[7*3+2])) * quarter;

    // Average Y direction
    T bx = ((pos[3*3+0] - pos[0*3+0]) + (pos[2*3+0] - pos[1*3+0])
          + (pos[7*3+0] - pos[4*3+0]) + (pos[6*3+0] - pos[5*3+0])) * quarter;
    T by = ((pos[3*3+1] - pos[0*3+1]) + (pos[2*3+1] - pos[1*3+1])
          + (pos[7*3+1] - pos[4*3+1]) + (pos[6*3+1] - pos[5*3+1])) * quarter;
    T bz = ((pos[3*3+2] - pos[0*3+2]) + (pos[2*3+2] - pos[1*3+2])
          + (pos[7*3+2] - pos[4*3+2]) + (pos[6*3+2] - pos[5*3+2])) * quarter;

    // Normalize X
    T invLen = myRsqrt(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    // Z = normalized cross(X, Y)
    T cx = ay * bz - az * by;
    T cy = az * bx - ax * bz;
    T cz = ax * by - ay * bx;
    invLen = myRsqrt(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    // Y = cross(Z, X)
    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

//=============================================================================
// Element data gathering
//=============================================================================

/// Gather positions for one element from global arrays (SoA layout)
template<typename T, int NNodes, int Dim>
__device__ inline void gatherElementData(
    const int* elements, int nbElem, int elemId,
    const T* globalData,
    T* localData)
{
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            localData[n * Dim + d] = globalData[nodeId * Dim + d];
    }
}

/// Gather displacement (x - x0) for one element
template<typename T, int NNodes, int Dim>
__device__ inline void gatherElementDisplacement(
    const int* elements, int nbElem, int elemId,
    const T* x, const T* x0,
    T* disp)
{
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            disp[n * Dim + d] = x[nodeId * Dim + d] - x0[nodeId * Dim + d];
    }
}

//=============================================================================
// Element center computation
//=============================================================================

/// Compute center of element positions
template<typename T, int NNodes, int Dim>
__device__ inline void computeElementCenter(const T* pos, T* center)
{
    const T invN = T(1) / T(NNodes);

    #pragma unroll
    for (int d = 0; d < Dim; ++d)
        center[d] = T(0);

    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            center[d] += pos[n * Dim + d];
    }

    #pragma unroll
    for (int d = 0; d < Dim; ++d)
        center[d] *= invN;
}

//=============================================================================
// Corotational displacement computation
//=============================================================================

/// Compute corotational displacement: disp = R^T * (x - center) - (x0 - center0)
template<typename T, int NNodes, int Dim>
__device__ inline void computeCorotationalDisplacement(
    const T* R,
    const T* x, const T* x0,
    const T* center, const T* center0,
    T* disp)
{
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        // diff = x_n - center
        T diff[Dim];
        #pragma unroll
        for (int d = 0; d < Dim; ++d)
            diff[d] = x[n * Dim + d] - center[d];

        // rotated = R^T * diff
        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T rotated = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                rotated += R[dj * Dim + di] * diff[dj];
            disp[n * Dim + di] = rotated - (x0[n * Dim + di] - center0[di]);
        }
    }
}

/// Compute R^T * dx for each node (for addDForce)
template<typename T, int NNodes, int Dim>
__device__ inline void rotateDisplacementTranspose(
    const T* R,
    const int* elements, int nbElem, int elemId,
    const T* dx,
    T* rdx)
{
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
}

//=============================================================================
// Symmetric block-matrix multiply
//=============================================================================

/**
 * Symmetric block-matrix multiply: out = K * in
 *
 * K is stored in upper-triangle block format:
 *   symIdx = ni * NNodes - ni*(ni-1)/2 + (nj - ni)  for nj >= ni
 *   K[symIdx * Dim * Dim + di * Dim + dj] for each element
 */
template<typename T, int NNodes, int Dim>
__device__ inline void symBlockMatMul(const T* K, const T* in, T* out)
{
    // Initialize output to zero
    #pragma unroll
    for (int i = 0; i < NNodes * Dim; ++i)
        out[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block: Kii * in_i -> out_i
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

        // Off-diagonal blocks (symmetric: Kij and Kij^T)
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

//=============================================================================
// Force output with rotation
//=============================================================================

/// Rotate local forces to global frame and write: out = scale * R * localForce
template<typename T, int NNodes, int Dim>
__device__ inline void rotateAndWriteForce(
    const T* R,
    const T* localForce,
    T* out,
    T scale)
{
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        #pragma unroll
        for (int di = 0; di < Dim; ++di)
        {
            T sum = T(0);
            #pragma unroll
            for (int dj = 0; dj < Dim; ++dj)
                sum += R[di * Dim + dj] * localForce[n * Dim + dj];
            out[n * Dim + di] = scale * sum;
        }
    }
}

/// Write negated force (for linear case without rotation): out = scale * localForce
template<typename T, int NNodes, int Dim>
__device__ inline void writeForce(const T* localForce, T* out, T scale)
{
    #pragma unroll
    for (int i = 0; i < NNodes * Dim; ++i)
        out[i] = scale * localForce[i];
}

//=============================================================================
// Gather kernel for accumulating per-vertex forces
//=============================================================================

/**
 * Gather per-vertex forces from per-element contributions.
 * velems[slot * nbVertex + vertexId] contains (elemId * NNodes + localNode + 1), 0 = end
 */
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
