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
 */
template<typename T, int NNodes>
__device__ void symBlockMatMul(const T* K, const T* in, T* out)
{
    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        out[i] = T(0);

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        {
            const T* Kii = K + diagIdx * 9;
            const T i0 = in[ni * 3 + 0];
            const T i1 = in[ni * 3 + 1];
            const T i2 = in[ni * 3 + 2];
            out[ni * 3 + 0] += Kii[0] * i0 + Kii[1] * i1 + Kii[2] * i2;
            out[ni * 3 + 1] += Kii[3] * i0 + Kii[4] * i1 + Kii[5] * i2;
            out[ni * 3 + 2] += Kii[6] * i0 + Kii[7] * i1 + Kii[8] * i2;
        }

        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const T* Kij = K + symIdx * 9;

            {
                const T j0 = in[nj * 3 + 0];
                const T j1 = in[nj * 3 + 1];
                const T j2 = in[nj * 3 + 2];
                out[ni * 3 + 0] += Kij[0] * j0 + Kij[1] * j1 + Kij[2] * j2;
                out[ni * 3 + 1] += Kij[3] * j0 + Kij[4] * j1 + Kij[5] * j2;
                out[ni * 3 + 2] += Kij[6] * j0 + Kij[7] * j1 + Kij[8] * j2;
            }

            {
                const T i0 = in[ni * 3 + 0];
                const T i1 = in[ni * 3 + 1];
                const T i2 = in[ni * 3 + 2];
                out[nj * 3 + 0] += Kij[0] * i0 + Kij[3] * i1 + Kij[6] * i2;
                out[nj * 3 + 1] += Kij[1] * i0 + Kij[4] * i1 + Kij[7] * i2;
                out[nj * 3 + 2] += Kij[2] * i0 + Kij[5] * i1 + Kij[8] * i2;
            }
        }
    }
}

/**
 * Combined kernel: compute rotations AND per-element forces in one pass.
 */
template<typename T, int NNodes>
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
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;
    const T invN = T(1) / T(NNodes);

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    T ex[NNodes * 3], ex0[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        ex[n * 3 + 0] = x[nodeId * 3 + 0];
        ex[n * 3 + 1] = x[nodeId * 3 + 1];
        ex[n * 3 + 2] = x[nodeId * 3 + 2];
        ex0[n * 3 + 0] = x0[nodeId * 3 + 0];
        ex0[n * 3 + 1] = x0[nodeId * 3 + 1];
        ex0[n * 3 + 2] = x0[nodeId * 3 + 2];
    }

    T frame[9];
    if constexpr (NNodes == 8)
        computeHexahedronFrame(ex, frame);
    else
        computeTriangleFrame(ex, frame);

    // R = frame^T * initRot
    const T* irt = initRotTransposed + elemId * 9;
    T R[9];
    mat3TransposeMul(frame, irt, R);

    T* Rout = rotationsOut + elemId * 9;
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        Rout[i] = R[i];

    T cx = T(0), cy = T(0), cz = T(0);
    T cx0 = T(0), cy0 = T(0), cz0 = T(0);
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        cx += ex[n * 3 + 0]; cy += ex[n * 3 + 1]; cz += ex[n * 3 + 2];
        cx0 += ex0[n * 3 + 0]; cy0 += ex0[n * 3 + 1]; cz0 += ex0[n * 3 + 2];
    }
    cx *= invN; cy *= invN; cz *= invN;
    cx0 *= invN; cy0 *= invN; cz0 *= invN;

    T disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const T dx = ex[n * 3 + 0] - cx;
        const T dy = ex[n * 3 + 1] - cy;
        const T dz = ex[n * 3 + 2] - cz;
        const T rx = R[0] * dx + R[3] * dy + R[6] * dz;
        const T ry = R[1] * dx + R[4] * dy + R[7] * dz;
        const T rz = R[2] * dx + R[5] * dy + R[8] * dz;
        disp[n * 3 + 0] = rx - (ex0[n * 3 + 0] - cx0);
        disp[n * 3 + 1] = ry - (ex0[n * 3 + 1] - cy0);
        disp[n * 3 + 2] = rz - (ex0[n * 3 + 2] - cz0);
    }

    T edf[NNodes * 3];
    const T* K = stiffness + elemId * NSymBlocks * 9;
    symBlockMatMul<T, NNodes>(K, disp, edf);

    T* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const T e0 = edf[n * 3 + 0];
        const T e1 = edf[n * 3 + 1];
        const T e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -(R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -(R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -(R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Kernel for addForce: Compute per-element force (1 thread per element).
 */
template<typename T, int NNodes>
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

    const T* Rptr = rotations + elemId * 9;
    T R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        R[i] = Rptr[i];

    T ex[NNodes * 3], ex0[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        ex[n * 3 + 0] = x[nodeId * 3 + 0];
        ex[n * 3 + 1] = x[nodeId * 3 + 1];
        ex[n * 3 + 2] = x[nodeId * 3 + 2];
        ex0[n * 3 + 0] = x0[nodeId * 3 + 0];
        ex0[n * 3 + 1] = x0[nodeId * 3 + 1];
        ex0[n * 3 + 2] = x0[nodeId * 3 + 2];
    }

    T cx = T(0), cy = T(0), cz = T(0);
    T cx0 = T(0), cy0 = T(0), cz0 = T(0);
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        cx += ex[n * 3 + 0]; cy += ex[n * 3 + 1]; cz += ex[n * 3 + 2];
        cx0 += ex0[n * 3 + 0]; cy0 += ex0[n * 3 + 1]; cz0 += ex0[n * 3 + 2];
    }
    cx *= invN; cy *= invN; cz *= invN;
    cx0 *= invN; cy0 *= invN; cz0 *= invN;

    T disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const T dx = ex[n * 3 + 0] - cx;
        const T dy = ex[n * 3 + 1] - cy;
        const T dz = ex[n * 3 + 2] - cz;
        const T rx = R[0] * dx + R[3] * dy + R[6] * dz;
        const T ry = R[1] * dx + R[4] * dy + R[7] * dz;
        const T rz = R[2] * dx + R[5] * dy + R[8] * dz;
        disp[n * 3 + 0] = rx - (ex0[n * 3 + 0] - cx0);
        disp[n * 3 + 1] = ry - (ex0[n * 3 + 1] - cy0);
        disp[n * 3 + 2] = rz - (ex0[n * 3 + 2] - cz0);
    }

    T edf[NNodes * 3];
    const T* K = stiffness + elemId * NSymBlocks * 9;
    symBlockMatMul<T, NNodes>(K, disp, edf);

    T* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const T e0 = edf[n * 3 + 0];
        const T e1 = edf[n * 3 + 1];
        const T e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -(R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -(R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -(R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Kernel for addDForce: Compute per-element dForce (1 thread per element).
 */
template<typename T, int NNodes>
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

    const T* Rptr = rotations + elemId * 9;
    T R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        R[i] = Rptr[i];

    T rdx[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const int nodeId = elements[n * nbElem + elemId];
        const T dx_x = dx[nodeId * 3 + 0];
        const T dx_y = dx[nodeId * 3 + 1];
        const T dx_z = dx[nodeId * 3 + 2];
        rdx[n * 3 + 0] = R[0] * dx_x + R[3] * dx_y + R[6] * dx_z;
        rdx[n * 3 + 1] = R[1] * dx_x + R[4] * dx_y + R[7] * dx_z;
        rdx[n * 3 + 2] = R[2] * dx_x + R[5] * dx_y + R[8] * dx_z;
    }

    const T* K = stiffness + elemId * NSymBlocks * 9;
    T edf[NNodes * 3];
    symBlockMatMul<T, NNodes>(K, rdx, edf);

    T* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const T e0 = edf[n * 3 + 0];
        const T e1 = edf[n * 3 + 1];
        const T e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -kFactor * (R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -kFactor * (R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -kFactor * (R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Gather per-vertex forces (1 thread per vertex).
 */
template<typename T>
__global__ void ElementCorotationalFEMForceField_gatherForce_kernel(
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
    ElementCorotationalFEMForceField_gatherForce_kernel<T>
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const T*)eforce,
            (T*)f);
    mycudaDebugError("ElementCorotationalFEMForceField_gatherForce_kernel");
}

template<typename T, int NNodes>
static void launchAddForceWithRotations(
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
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeRotationsAndForce_kernel<T, NNodes>
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

    launchGather<T>(nbVertex, maxElemPerVertex, velems, eforce, f);
}

template<typename T, int NNodes>
static void launchAddForce(
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
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeForce_kernel<T, NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)rotations,
            (const T*)stiffness,
            (const T*)x,
            (const T*)x0,
            (T*)eforce);
    mycudaDebugError("ElementCorotationalFEMForceField_computeForce_kernel");

    launchGather<T>(nbVertex, maxElemPerVertex, velems, eforce, f);
}

template<typename T, int NNodes>
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
    T kFactor)
{
    const int computeThreads = 64;
    const int numBlocks = (nbElem + computeThreads - 1) / computeThreads;
    ElementCorotationalFEMForceField_computeDForce_kernel<T, NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const T*)rotations,
            (const T*)stiffness,
            (const T*)dx,
            (T*)eforce,
            kFactor);
    mycudaDebugError("ElementCorotationalFEMForceField_computeDForce_kernel");

    launchGather<T>(nbVertex, maxElemPerVertex, velems, eforce, df);
}

extern "C"
{

// ==================== float versions ====================

void ElementCorotationalFEMForceFieldCuda3f_addForceWithRotations(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
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
    switch (nbNodesPerElem)
    {
        case 3: launchAddForceWithRotations<float, 3>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 4: launchAddForceWithRotations<float, 4>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 8: launchAddForceWithRotations<float, 8>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
    }
}

void ElementCorotationalFEMForceFieldCuda3f_addForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
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
    switch (nbNodesPerElem)
    {
        case 2: launchAddForce<float, 2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<float, 3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<float, 4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<float, 8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
    }
}

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
        case 2: launchAddDForce<float, 2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<float, 3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<float, 4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<float, 8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

// ==================== double versions ====================

void ElementCorotationalFEMForceFieldCuda3d_addForceWithRotations(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
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
    switch (nbNodesPerElem)
    {
        case 3: launchAddForceWithRotations<double, 3>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 4: launchAddForceWithRotations<double, 4>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 8: launchAddForceWithRotations<double, 8>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
    }
}

void ElementCorotationalFEMForceFieldCuda3d_addForce(
    unsigned int nbElem,
    unsigned int nbVertex,
    unsigned int nbNodesPerElem,
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
    switch (nbNodesPerElem)
    {
        case 2: launchAddForce<double, 2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<double, 3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<double, 4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<double, 8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
    }
}

void ElementCorotationalFEMForceFieldCuda3d_addDForce(
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
    double kFactor)
{
    switch (nbNodesPerElem)
    {
        case 2: launchAddDForce<double, 2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 3: launchAddDForce<double, 3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 4: launchAddDForce<double, 4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
        case 8: launchAddDForce<double, 8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, dx, df, eforce, velems, kFactor); break;
    }
}

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
