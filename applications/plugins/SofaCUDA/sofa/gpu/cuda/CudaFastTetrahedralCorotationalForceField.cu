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
#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"


#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

struct GPUTetrahedron
{
    int indices[4];
    int edgeIndices[6];
};

struct GPUEdge
{
    int indices[2];
};

template <typename real>
struct GPUTetrahedronRestInformation
{
    CudaVec3<real> shapeVector[4];
    real restVolume;
    CudaVec3<real> restEdgeVector[6];

    matrix3<real> linearDfDxDiag[4];  // the diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
    matrix3<real> linearDfDx[6];  // the off-diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
    matrix3<real> rotation; // rotation from deformed to rest configuration
    matrix3<real> restRotation; // used for QR decomposition

    real edgeOrientation[6];
};


//////////////////////
// GPU-side methods //
//////////////////////

template <typename real>
__global__ void FastTetrahedralCorotationalForceFieldCudaVec3_addForce_kernel(int size, CudaVec3<real>* f, const CudaVec3<real>* x, const CudaVec3<real>* v,
    unsigned int nbTetrahedra, GPUTetrahedronRestInformation<real>* tetrahedronInfo, const GPUTetrahedron* gpuTetra)
{
    using CudaVec3 = CudaVec3<real>;
    int index0 = (blockIdx.x*BSIZE);
    int index = threadIdx.x;
    int tetraId = index0+index;

    GPUTetrahedron tetra = gpuTetra[tetraId];
    GPUTetrahedronRestInformation<real>& tetraRInfo = tetrahedronInfo[tetraId];
    
    // compute current tetrahedron displacement
    const unsigned int edgesInTetrahedronArray[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} }; // TODO check how to store static array on device
    CudaVec3 displ[6];
    for (int j = 0; j < 6; ++j)
    {
        displ[j] = x[tetra.indices[edgesInTetrahedronArray[j][1]]] - x[tetra.indices[edgesInTetrahedronArray[j][0]]];
    }

    // Only QR_DECOMPOSITION for the moment    
    /// perform QR decomposition
    //computeQRRotation(rot, displ);
    matrix3<real> frame;
    CudaVec3 edgex = displ[0];
    CudaVec3 edgey = displ[1];
    CudaVec3 edgez = cross(edgex, edgey);
    edgey = cross(edgez, edgex);

    edgex *= invnorm(edgex);
    edgey *= invnorm(edgey);
    edgez *= invnorm(edgez);

    frame = matrix3<real>::make(edgex.x, edgex.y, edgex.z, edgey.x, edgey.y, edgey.z, edgez.x, edgez.y, edgez.z);
    matrix3<real> rot = frame.transpose(frame) * tetraRInfo.restRotation;

    // store transpose of rotation
    matrix3<real> rotT = rot.transpose(rot);
    tetraRInfo.rotation = rotT;
    
    CudaVec3 force[4];
    force[0] = CudaVec3::make(0.0, 0.0, 0.0);
    force[1] = CudaVec3::make(0.0, 0.0, 0.0);
    force[2] = CudaVec3::make(0.0, 0.0, 0.0);
    force[3] = CudaVec3::make(0.0, 0.0, 0.0);

    for (int j = 0; j < 6; ++j)
    {
        // displacement in the rest configuration
        displ[j] = tetraRInfo.rotation * displ[j] - tetraRInfo.restEdgeVector[j];

        // force on first vertex in the rest configuration
        force[edgesInTetrahedronArray[j][1]] += tetraRInfo.linearDfDx[j] * displ[j];
        
        // force on second vertex in the rest configuration
        matrix3<real> linearT = tetraRInfo.linearDfDx[j].transpose(tetraRInfo.linearDfDx[j]);
        force[edgesInTetrahedronArray[j][0]] -= linearT * displ[j];
    }

    for (int j = 0; j < 4; ++j)
    {
        force[j] = rot * force[j];
        unsigned int idV = tetra.indices[j];

        atomicAdd(&(f[idV].x), force[j].x);
        atomicAdd(&(f[idV].y), force[j].y);
        atomicAdd(&(f[idV].z), force[j].z);
    }
}


template <typename real>
__global__ void FastTetrahedralCorotationalForceFieldCudaVec3_computeEdgeMatrices_kernel(unsigned int nbTetrahedra,
    const GPUTetrahedronRestInformation<real>* tetrahedronInfo, matrix3<real>* edgeDfDx, const GPUTetrahedron* gpuTetra)
{
    int tetraId = (blockIdx.x * BSIZE) + threadIdx.x;

    GPUTetrahedron tetra = gpuTetra[tetraId];
    const GPUTetrahedronRestInformation<real>& tetraRInfo = tetrahedronInfo[tetraId];

    for (unsigned int i = 0; i < 6; ++i)
    {
        unsigned int edgeID = tetra.edgeIndices[i];

        // test if the tetrahedron edge has the same orientation as the global edge
        matrix3<real> tmp = tetraRInfo.linearDfDx[i] * tetraRInfo.rotation;
        matrix3<real> edgeMatrix;
        if (tetraRInfo.edgeOrientation[i] == 1)
        {
            // store the two edge matrices since the stiffness matrix is not symmetric
            matrix3<real> rot = tetraRInfo.rotation;
            edgeMatrix = rot.transpose(rot) * tmp;
        }
        else
        {
            edgeMatrix = tmp.transpose(tmp) * tetraRInfo.rotation;
        }

        matrix3<real>& eDfDx = edgeDfDx[edgeID];
        
        atomicAdd(&eDfDx.x.x, edgeMatrix.x.x);
        atomicAdd(&eDfDx.x.y, edgeMatrix.x.y);
        atomicAdd(&eDfDx.x.z, edgeMatrix.x.z);

        atomicAdd(&eDfDx.y.x, edgeMatrix.y.x);
        atomicAdd(&eDfDx.y.y, edgeMatrix.y.y);
        atomicAdd(&eDfDx.y.z, edgeMatrix.y.z);

        atomicAdd(&eDfDx.z.x, edgeMatrix.z.x);
        atomicAdd(&eDfDx.z.y, edgeMatrix.z.y);
        atomicAdd(&eDfDx.z.z, edgeMatrix.z.z);
    }
}

template <typename real>
__global__ void FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel(unsigned int nbedges, CudaVec3<real>* df, const CudaVec3<real>* dx, real kFactor,
    const matrix3<real>* edgeDfDx, const GPUEdge* gpuEdges)
{
    int edgeId = (blockIdx.x * BSIZE) + threadIdx.x;
    GPUEdge edge = gpuEdges[edgeId];
    matrix3<real> eDfDx = edgeDfDx[edgeId];

    CudaVec3<real> deltax = dx[edge.indices[1]] - dx[edge.indices[0]];
    deltax *= kFactor;

    // use the already stored matrix
    matrix3<real> eDfDxT = eDfDx.transpose(eDfDx);
    CudaVec3<real> df0 = eDfDxT * deltax;

    CudaVec3<real> df1 = eDfDx * deltax;

    atomicAdd(&(df[edge.indices[0]].x), -df0.x);
    atomicAdd(&(df[edge.indices[0]].y), -df0.y);
    atomicAdd(&(df[edge.indices[0]].z), -df0.z);

    atomicAdd(&(df[edge.indices[1]].x), df1.x);
    atomicAdd(&(df[edge.indices[1]].y), df1.y);
    atomicAdd(&(df[edge.indices[1]].z), df1.z);
}


//////////////////////
// CPU-side methods //
//////////////////////

extern "C"
{

void FastTetrahedralCorotationalForceFieldCuda3f_addForce(unsigned int size, void* f, const void* x, const void* v,
    unsigned int nbTetrahedra, void* tetrahedronInfo, const void* gpuTetra)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTetrahedra + BSIZE - 1) / BSIZE, 1);
    {
        FastTetrahedralCorotationalForceFieldCudaVec3_addForce_kernel<float> <<< grid, threads >>> (size, (CudaVec3f*)f, (const CudaVec3f*)x, (const CudaVec3f*)v,
            nbTetrahedra, (GPUTetrahedronRestInformation<float>*)tetrahedronInfo, (const GPUTetrahedron*)gpuTetra);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addForce_kernel"); 
    }
}

void FastTetrahedralCorotationalForceFieldCuda3f_computeEdgeMatrices(unsigned int nbTetrahedra, const void* tetrahedronInfo, void* edgeDfDx, const void* gpuTetra)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTetrahedra + BSIZE - 1) / BSIZE, 1);
    {        
        FastTetrahedralCorotationalForceFieldCudaVec3_computeEdgeMatrices_kernel<float> <<< grid, threads >>> (nbTetrahedra,
            (const GPUTetrahedronRestInformation<float>*)tetrahedronInfo, (matrix3<float>*)edgeDfDx,(const GPUTetrahedron*)gpuTetra);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addForce_kernel"); 
    }
}

void FastTetrahedralCorotationalForceFieldCuda3f_addDForce(unsigned int nbedges, void* df, const void* dx, float kFactor,
    const void* edgeDfDx, const void* gpuEdges)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbedges + BSIZE - 1) / BSIZE, 1);
    {
        FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel<float> <<< grid, threads >>> (nbedges, (CudaVec3f*)df, (const CudaVec3f*)dx, kFactor,
            (const matrix3<float>*)edgeDfDx, (const GPUEdge*)gpuEdges);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addDForce_kernel"); 
    }
}


#ifdef SOFA_GPU_CUDA_DOUBLE
void FastTetrahedralCorotationalForceFieldCuda3d_addForce(unsigned int size, void* f, const void* x, const void* v,
    unsigned int nbTetrahedra, void* tetrahedronInfo, const void* gpuTetra)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTetrahedra + BSIZE - 1) / BSIZE, 1);
    {
        FastTetrahedralCorotationalForceFieldCudaVec3_addForce_kernel<double> << < grid, threads >> > (size, (CudaVec3d*)f, (const CudaVec3d*)x, (const CudaVec3d*)v,
            nbTetrahedra, (GPUTetrahedronRestInformation<double>*)tetrahedronInfo, (const GPUTetrahedron*)gpuTetra);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3d_addForce_kernel"); }
}

void FastTetrahedralCorotationalForceFieldCuda3d_computeEdgeMatrices(unsigned int nbTetrahedra, const void* tetrahedronInfo, void* edgeDfDx, const void* gpuTetra)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTetrahedra + BSIZE - 1) / BSIZE, 1);
    {
        FastTetrahedralCorotationalForceFieldCudaVec3_computeEdgeMatrices_kernel<double> << < grid, threads >> > (nbTetrahedra,
            (const GPUTetrahedronRestInformation<double>*)tetrahedronInfo, (matrix3<double>*)edgeDfDx, (const GPUTetrahedron*)gpuTetra);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3d_computeEdgeMatrices_kernel");
    }
}

void FastTetrahedralCorotationalForceFieldCuda3d_addDForce(unsigned int nbedges, void* df, const void* dx, float kFactor,
    const void* edgeDfDx, const void* gpuEdges)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbedges + BSIZE - 1) / BSIZE, 1);
    {
        FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel<double> << < grid, threads >> > (nbedges, (CudaVec3d*)df, (const CudaVec3d*)dx, kFactor,
            (const matrix3<double>*)edgeDfDx, (const GPUEdge*)gpuEdges);
        mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3d_addDForce_kernel");
    }
}
#endif

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
