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

/// Maximum number of DOFs per element (8 nodes * 3 dimensions for hexahedra)
#define MAX_ELEM_DOFS 24
/// Maximum spatial dimensions
#define MAX_DIM 3
/// Maximum nodes per element
#define MAX_NODES 8

/**
 * CUDA kernel for addDForce of corotational FEM.
 *
 * Generic over element type: works with any number of nodes per element and spatial dimensions.
 * One thread per element. For each element:
 *   1. Gather dx from nodes
 *   2. Rotate dx into reference frame: rdx = R^T * dx
 *   3. Multiply by stiffness: edf = K * rdx
 *   4. Rotate back: df_world = R * edf
 *   5. Scatter to nodes via atomicAdd: df[node] -= kFactor * df_world
 */
__global__ void ElementCorotationalFEMForceFieldCuda3f_addDForce_kernel(
    int nbElem,
    int nbNodesPerElem,
    int nbDofsPerElem,
    int dim,
    const int* __restrict__ elements,
    const float* __restrict__ rotations,
    const float* __restrict__ stiffness,
    const float* __restrict__ dx,
    float* df,
    float kFactor)
{
    const int elemId = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemId >= nbElem) return;

    // Load element node indices
    const int* elemNodes = elements + elemId * nbNodesPerElem;

    // Load rotation matrix R (dim x dim, row-major)
    const float* Rptr = rotations + elemId * dim * dim;
    float R[MAX_DIM * MAX_DIM];
    for (int i = 0; i < dim * dim; ++i)
        R[i] = Rptr[i];

    // Gather dx and rotate into reference frame: rdx = R^T * dx_node
    float rdx[MAX_ELEM_DOFS];
    for (int n = 0; n < nbNodesPerElem; ++n)
    {
        const int nodeId = elemNodes[n];
        const float* node_dx = dx + nodeId * dim;

        for (int i = 0; i < dim; ++i)
        {
            float val = 0.0f;
            for (int j = 0; j < dim; ++j)
                val += R[j * dim + i] * node_dx[j]; // R^T[i][j] = R[j][i]
            rdx[n * dim + i] = val;
        }
    }

    // K * rdx -> edf (nbDofsPerElem x nbDofsPerElem matrix-vector product)
    const float* K = stiffness + elemId * nbDofsPerElem * nbDofsPerElem;
    float edf[MAX_ELEM_DOFS];
    for (int i = 0; i < nbDofsPerElem; ++i)
    {
        float sum = 0.0f;
        const float* Ki = K + i * nbDofsPerElem;
        for (int j = 0; j < nbDofsPerElem; ++j)
            sum += Ki[j] * rdx[j];
        edf[i] = sum;
    }

    // Rotate back and scatter: df[node] -= kFactor * R * edf_node
    for (int n = 0; n < nbNodesPerElem; ++n)
    {
        const int nodeId = elemNodes[n];
        const float* node_edf = edf + n * dim;

        for (int i = 0; i < dim; ++i)
        {
            float val = 0.0f;
            for (int j = 0; j < dim; ++j)
                val += R[i * dim + j] * node_edf[j]; // R * edf_node
            atomicAdd(&df[nodeId * dim + i], -kFactor * val);
        }
    }
}

extern "C"
{

void ElementCorotationalFEMForceFieldCuda3f_addDForce(
    unsigned int nbElem,
    unsigned int nbNodesPerElem,
    unsigned int nbDofsPerElem,
    unsigned int spatialDim,
    const void* elements,
    const void* rotations,
    const void* stiffness,
    const void* dx,
    void* df,
    float kFactor)
{
    const int threadsPerBlock = 64;
    const int numBlocks = (nbElem + threadsPerBlock - 1) / threadsPerBlock;

    ElementCorotationalFEMForceFieldCuda3f_addDForce_kernel<<<numBlocks, threadsPerBlock>>>(
        nbElem,
        nbNodesPerElem,
        nbDofsPerElem,
        spatialDim,
        (const int*)elements,
        (const float*)rotations,
        (const float*)stiffness,
        (const float*)dx,
        (float*)df,
        kFactor);

    mycudaDebugError("ElementCorotationalFEMForceFieldCuda3f_addDForce_kernel");
}

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
