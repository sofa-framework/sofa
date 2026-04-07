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
 * Device helper: 3x3 matrix multiply C = A * B (row-major)
 */
__device__ void mat3Mul(const float* A, const float* B, float* C)
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
__device__ void mat3MulTranspose(const float* A, const float* BT, float* C)
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
 * Device helper: compute rotation frame from first 3 nodes (TriangleRotation).
 * Used for Triangle (NNodes=3) and Tetrahedron (NNodes=4) elements.
 * ex is [NNodes*3] array of gathered node positions.
 */
__device__ void computeTriangleFrame(const float* ex, float* frame)
{
    // xAxis = normalize(p1 - p0)
    float ax = ex[3] - ex[0], ay = ex[4] - ex[1], az = ex[5] - ex[2];
    float invLen = rsqrtf(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    // tmp yAxis = p2 - p0
    float bx = ex[6] - ex[0], by = ex[7] - ex[1], bz = ex[8] - ex[2];

    // zAxis = normalize(cross(xAxis, tmpY))
    float cx = ay * bz - az * by;
    float cy = az * bx - ax * bz;
    float cz = ax * by - ay * bx;
    invLen = rsqrtf(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    // yAxis = cross(zAxis, xAxis)
    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    // frame rows: [xAxis; yAxis; zAxis]
    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

/**
 * Device helper: compute rotation frame from 8 hexahedron nodes (HexahedronRotation).
 * ex is [8*3] array of gathered node positions.
 */
__device__ void computeHexahedronFrame(const float* ex, float* frame)
{
    // Average edge vectors
    // xAxis_avg = ((n1-n0) + (n2-n3) + (n5-n4) + (n6-n7)) * 0.25
    float ax = ((ex[1*3+0] - ex[0*3+0]) + (ex[2*3+0] - ex[3*3+0])
              + (ex[5*3+0] - ex[4*3+0]) + (ex[6*3+0] - ex[7*3+0])) * 0.25f;
    float ay = ((ex[1*3+1] - ex[0*3+1]) + (ex[2*3+1] - ex[3*3+1])
              + (ex[5*3+1] - ex[4*3+1]) + (ex[6*3+1] - ex[7*3+1])) * 0.25f;
    float az = ((ex[1*3+2] - ex[0*3+2]) + (ex[2*3+2] - ex[3*3+2])
              + (ex[5*3+2] - ex[4*3+2]) + (ex[6*3+2] - ex[7*3+2])) * 0.25f;

    // yAxis_avg = ((n3-n0) + (n2-n1) + (n7-n4) + (n6-n5)) * 0.25
    float bx = ((ex[3*3+0] - ex[0*3+0]) + (ex[2*3+0] - ex[1*3+0])
              + (ex[7*3+0] - ex[4*3+0]) + (ex[6*3+0] - ex[5*3+0])) * 0.25f;
    float by = ((ex[3*3+1] - ex[0*3+1]) + (ex[2*3+1] - ex[1*3+1])
              + (ex[7*3+1] - ex[4*3+1]) + (ex[6*3+1] - ex[5*3+1])) * 0.25f;
    float bz = ((ex[3*3+2] - ex[0*3+2]) + (ex[2*3+2] - ex[1*3+2])
              + (ex[7*3+2] - ex[4*3+2]) + (ex[6*3+2] - ex[5*3+2])) * 0.25f;

    // Normalize xAxis
    float invLen = rsqrtf(ax * ax + ay * ay + az * az);
    ax *= invLen; ay *= invLen; az *= invLen;

    // zAxis = normalize(cross(xAxis, yAxis_avg))
    float cx = ay * bz - az * by;
    float cy = az * bx - ax * bz;
    float cz = ax * by - ay * bx;
    invLen = rsqrtf(cx * cx + cy * cy + cz * cz);
    cx *= invLen; cy *= invLen; cz *= invLen;

    // yAxis = cross(zAxis, xAxis)
    bx = cy * az - cz * ay;
    by = cz * ax - cx * az;
    bz = cx * ay - cy * ax;

    // frame rows: [xAxis; yAxis; zAxis]
    frame[0] = ax; frame[1] = ay; frame[2] = az;
    frame[3] = bx; frame[4] = by; frame[5] = bz;
    frame[6] = cx; frame[7] = cy; frame[8] = cz;
}

/**
 * Symmetric block-matrix multiply: out = K * in
 * K stored as upper triangle: NSymBlocks = NNodes*(NNodes+1)/2 blocks of 9 floats.
 * Inline device function shared by addForce, addDForce, and combined kernels.
 */
template<int NNodes>
__device__ void symBlockMatMul(const float* K, const float* in, float* out)
{
    #pragma unroll
    for (int i = 0; i < NNodes * 3; ++i)
        out[i] = 0.0f;

    #pragma unroll
    for (int ni = 0; ni < NNodes; ++ni)
    {
        const int diagIdx = ni * NNodes - ni * (ni - 1) / 2;

        // Diagonal block
        {
            const float* Kii = K + diagIdx * 9;
            const float i0 = in[ni * 3 + 0];
            const float i1 = in[ni * 3 + 1];
            const float i2 = in[ni * 3 + 2];
            out[ni * 3 + 0] += Kii[0] * i0 + Kii[1] * i1 + Kii[2] * i2;
            out[ni * 3 + 1] += Kii[3] * i0 + Kii[4] * i1 + Kii[5] * i2;
            out[ni * 3 + 2] += Kii[6] * i0 + Kii[7] * i1 + Kii[8] * i2;
        }

        // Off-diagonal blocks
        #pragma unroll
        for (int nj = ni + 1; nj < NNodes; ++nj)
        {
            const int symIdx = diagIdx + (nj - ni);
            const float* Kij = K + symIdx * 9;

            // Forward: out[ni] += Kij * in[nj]
            {
                const float j0 = in[nj * 3 + 0];
                const float j1 = in[nj * 3 + 1];
                const float j2 = in[nj * 3 + 2];
                out[ni * 3 + 0] += Kij[0] * j0 + Kij[1] * j1 + Kij[2] * j2;
                out[ni * 3 + 1] += Kij[3] * j0 + Kij[4] * j1 + Kij[5] * j2;
                out[ni * 3 + 2] += Kij[6] * j0 + Kij[7] * j1 + Kij[8] * j2;
            }

            // Symmetric: out[nj] += Kij^T * in[ni]
            {
                const float i0 = in[ni * 3 + 0];
                const float i1 = in[ni * 3 + 1];
                const float i2 = in[ni * 3 + 2];
                out[nj * 3 + 0] += Kij[0] * i0 + Kij[3] * i1 + Kij[6] * i2;
                out[nj * 3 + 1] += Kij[1] * i0 + Kij[4] * i1 + Kij[7] * i2;
                out[nj * 3 + 2] += Kij[2] * i0 + Kij[5] * i1 + Kij[8] * i2;
            }
        }
    }
}

/**
 * Combined kernel: compute rotations AND per-element forces in one pass.
 *
 * Uses TriangleRotation for NNodes=3,4 and HexahedronRotation for NNodes=8.
 * Computes: frame from node positions → R = frame * initRotTransposed
 * Then: displacement = R^T*(x-centroid) - (x0-centroid0) → K*disp → -R*result
 * Also writes R to rotations buffer for subsequent addDForce calls.
 */
template<int NNodes>
__global__ void ElementCorotationalFEMForceFieldCuda3f_computeRotationsAndForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const float* __restrict__ initRotTransposed,
    const float* __restrict__ stiffness,
    const float* __restrict__ x,
    const float* __restrict__ x0,
    float* __restrict__ rotationsOut,
    float* __restrict__ eforce)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;
    constexpr float invN = 1.0f / NNodes;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Gather node positions and rest positions
    float ex[NNodes * 3], ex0[NNodes * 3];
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

    // Compute rotation frame from current positions
    float frame[9];
    if constexpr (NNodes == 8)
        computeHexahedronFrame(ex, frame);
    else
        computeTriangleFrame(ex, frame);

    // R = frame * initRotTransposed^T (i.e. frame.multTranspose(initRotTransposed))
    // Since initRotTransposed is already the transpose, R = frame * initRotTransposed^T
    const float* irt = initRotTransposed + elemId * 9;
    float R[9];
    mat3MulTranspose(frame, irt, R);

    // Write R to rotations buffer for addDForce
    float* Rout = rotationsOut + elemId * 9;
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        Rout[i] = R[i];

    // Compute centroids
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    float cx0 = 0.0f, cy0 = 0.0f, cz0 = 0.0f;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        cx += ex[n * 3 + 0]; cy += ex[n * 3 + 1]; cz += ex[n * 3 + 2];
        cx0 += ex0[n * 3 + 0]; cy0 += ex0[n * 3 + 1]; cz0 += ex0[n * 3 + 2];
    }
    cx *= invN; cy *= invN; cz *= invN;
    cx0 *= invN; cy0 *= invN; cz0 *= invN;

    // Compute displacement: disp[j] = R^T * (x[j] - centroid) - (x0[j] - centroid0)
    float disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const float dx = ex[n * 3 + 0] - cx;
        const float dy = ex[n * 3 + 1] - cy;
        const float dz = ex[n * 3 + 2] - cz;
        const float rx = R[0] * dx + R[3] * dy + R[6] * dz;
        const float ry = R[1] * dx + R[4] * dy + R[7] * dz;
        const float rz = R[2] * dx + R[5] * dy + R[8] * dz;
        disp[n * 3 + 0] = rx - (ex0[n * 3 + 0] - cx0);
        disp[n * 3 + 1] = ry - (ex0[n * 3 + 1] - cy0);
        disp[n * 3 + 2] = rz - (ex0[n * 3 + 2] - cz0);
    }

    // edf = K * disp
    float edf[NNodes * 3];
    const float* K = stiffness + elemId * NSymBlocks * 9;
    symBlockMatMul<NNodes>(K, disp, edf);

    // Rotate back and write: out = -R * edf
    float* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const float e0 = edf[n * 3 + 0];
        const float e1 = edf[n * 3 + 1];
        const float e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -(R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -(R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -(R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Kernel for addForce: Compute per-element force (1 thread per element).
 *
 * displacement[j] = R^T * (x[j] - centroid_x) - (x0[j] - centroid_x0)
 * elementForce = K * displacement
 * out[j] = -R * elementForce[j]
 */
template<int NNodes>
__global__ void ElementCorotationalFEMForceFieldCuda3f_computeForce_kernel(
    int nbElem,
    const int* __restrict__ elements,
    const float* __restrict__ rotations,
    const float* __restrict__ stiffness,
    const float* __restrict__ x,
    const float* __restrict__ x0,
    float* __restrict__ eforce)
{
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;
    constexpr float invN = 1.0f / NNodes;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load rotation matrix R (3x3, row-major)
    const float* Rptr = rotations + elemId * 9;
    float R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        R[i] = Rptr[i];

    // Gather node positions and rest positions
    float ex[NNodes * 3], ex0[NNodes * 3];
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

    // Compute centroids
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    float cx0 = 0.0f, cy0 = 0.0f, cz0 = 0.0f;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        cx += ex[n * 3 + 0]; cy += ex[n * 3 + 1]; cz += ex[n * 3 + 2];
        cx0 += ex0[n * 3 + 0]; cy0 += ex0[n * 3 + 1]; cz0 += ex0[n * 3 + 2];
    }
    cx *= invN; cy *= invN; cz *= invN;
    cx0 *= invN; cy0 *= invN; cz0 *= invN;

    // Compute displacement: disp[j] = R^T * (x[j] - centroid) - (x0[j] - centroid0)
    float disp[NNodes * 3];
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const float dx = ex[n * 3 + 0] - cx;
        const float dy = ex[n * 3 + 1] - cy;
        const float dz = ex[n * 3 + 2] - cz;

        // R^T * (x - centroid)
        const float rx = R[0] * dx + R[3] * dy + R[6] * dz;
        const float ry = R[1] * dx + R[4] * dy + R[7] * dz;
        const float rz = R[2] * dx + R[5] * dy + R[8] * dz;

        disp[n * 3 + 0] = rx - (ex0[n * 3 + 0] - cx0);
        disp[n * 3 + 1] = ry - (ex0[n * 3 + 1] - cy0);
        disp[n * 3 + 2] = rz - (ex0[n * 3 + 2] - cz0);
    }

    // edf = K * disp
    float edf[NNodes * 3];
    const float* K = stiffness + elemId * NSymBlocks * 9;
    symBlockMatMul<NNodes>(K, disp, edf);

    // Rotate back and write: out = -R * edf
    float* out = eforce + elemId * NNodes * 3;
    #pragma unroll
    for (int n = 0; n < NNodes; ++n)
    {
        const float e0 = edf[n * 3 + 0];
        const float e1 = edf[n * 3 + 1];
        const float e2 = edf[n * 3 + 2];
        out[n * 3 + 0] = -(R[0] * e0 + R[1] * e1 + R[2] * e2);
        out[n * 3 + 1] = -(R[3] * e0 + R[4] * e1 + R[5] * e2);
        out[n * 3 + 2] = -(R[6] * e0 + R[7] * e1 + R[8] * e2);
    }
}

/**
 * Kernel for addDForce: Compute per-element dForce (1 thread per element).
 *
 * rdx = R^T * dx, edf = K * rdx, out = -kFactor * R * edf
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
    constexpr int NSymBlocks = NNodes * (NNodes + 1) / 2;

    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load rotation matrix R (3x3, row-major)
    const float* Rptr = rotations + elemId * 9;
    float R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i)
        R[i] = Rptr[i];

    // Gather dx and rotate into reference frame: rdx[n] = R^T * dx[node[n]]
    float rdx[NNodes * 3];
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

    // Symmetric block-matrix multiply: edf = K * rdx
    const float* K = stiffness + elemId * NSymBlocks * 9;
    float edf[NNodes * 3];
    symBlockMatMul<NNodes>(K, rdx, edf);

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
 * Gather per-vertex forces (1 thread per vertex).
 *
 * Shared by addForce and addDForce.
 * No atomics: each vertex handled by exactly one thread.
 * velems is SoA: velems[s * nbVertex + vertexId], 0-terminated.
 * Each entry is (elemId * NNodes + localNode + 1), with 0 as sentinel.
 */
__global__ void ElementCorotationalFEMForceFieldCuda3f_gatherForce_kernel(
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
    ElementCorotationalFEMForceFieldCuda3f_gatherForce_kernel
        <<<numBlocks, gatherThreads>>>(
            nbVertex,
            maxElemPerVertex,
            (const int*)velems,
            (const float*)eforce,
            (float*)f);
    mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_gatherForce_kernel");
}

template<int NNodes>
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
    ElementCorotationalFEMForceFieldCuda3f_computeRotationsAndForce_kernel<NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const float*)initRotTransposed,
            (const float*)stiffness,
            (const float*)x,
            (const float*)x0,
            (float*)rotationsOut,
            (float*)eforce);
    mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_computeRotationsAndForce_kernel");

    launchGather(nbVertex, maxElemPerVertex, velems, eforce, f);
}

template<int NNodes>
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
    ElementCorotationalFEMForceFieldCuda3f_computeForce_kernel<NNodes>
        <<<numBlocks, computeThreads>>>(
            nbElem,
            (const int*)elements,
            (const float*)rotations,
            (const float*)stiffness,
            (const float*)x,
            (const float*)x0,
            (float*)eforce);
    mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_computeForce_kernel");

    launchGather(nbVertex, maxElemPerVertex, velems, eforce, f);
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

    launchGather(nbVertex, maxElemPerVertex, velems, eforce, df);
}

extern "C"
{

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
        case 3: launchAddForceWithRotations<3>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 4: launchAddForceWithRotations<4>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
        case 8: launchAddForceWithRotations<8>(nbElem, nbVertex, maxElemPerVertex, elements, initRotTransposed, stiffness, x, x0, f, eforce, rotationsOut, velems); break;
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
        case 2: launchAddForce<2>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 3: launchAddForce<3>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 4: launchAddForce<4>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
        case 8: launchAddForce<8>(nbElem, nbVertex, maxElemPerVertex, elements, rotations, stiffness, x, x0, f, eforce, velems); break;
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
