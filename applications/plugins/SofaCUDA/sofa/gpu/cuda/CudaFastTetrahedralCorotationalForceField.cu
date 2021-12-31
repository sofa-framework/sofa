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


//template <typename real>
//__global__ void FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel(int size, CudaVec3<real>* df, const CudaVec3<real>* dx, real kFactor,
//    const TriangleState<real>* triState, const TriangleInfo<real>* triInfo,
//    unsigned int nbTriangles,
//    const GPUTriangleInfo* gpuTriangleInfo,
//    real gamma, real mu
//)
//{
//    using CudaVec3 = CudaVec3<real>;
//    int index0 = (blockIdx.x*BSIZE);
//    int index = threadIdx.x;
//    int i = index0+index;
//    
//    GPUTriangleInfo t = gpuTriangleInfo[i];
//    const TriangleInfo<real>& ti = triInfo[i];
//    const TriangleState<real>& ts = triState[i];
//
//    CudaVec3 da  = dx[t.ia];
//    CudaVec3 dab = dx[t.ib]-da;
//    CudaVec3 dac = dx[t.ic]-da;
//    real dbx = dot(ts.frame_x, dab);
//    real dby = dot(ts.frame_y, dab);
//    real dcx = dot(ts.frame_x, dac);
//    real dcy = dot(ts.frame_y, dac);
//
//    CudaVec3 dstrain = CudaVec3::make (
//        ti.cy  * dbx,                             // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
//        ti.bx * dcy - ti.cx * dby,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
//        ti.bx * dcx - ti.cx * dbx + ti.cy * dby); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)
//
//    real gammaXY = (dstrain.x + dstrain.y) * gamma;
//
//    CudaVec3 dstress = CudaVec3::make (
//        mu*dstrain.x + gammaXY,    // (gamma+mu, gamma   ,    0) * dstrain
//        mu*dstrain.y + gammaXY,    // (gamma   , gamma+mu,    0) * dstrain
//        (float)(0.5)*mu*dstrain.z); // (       0,        0, mu/2) * dstrain
//
//    dstress *= ti.ss_factor * kFactor;
//    CudaVec3 dfb = ts.frame_x * (ti.cy * dstress.x - ti.cx * dstress.z)  // (cy,   0, -cx) * dstress
//            + ts.frame_y * (ti.cy * dstress.z - ti.cx * dstress.y);   // ( 0, -cx,  cy) * dstress
//    CudaVec3 dfc = ts.frame_x * (ti.bx * dstress.z)                      // ( 0,   0,  bx) * dstress
//            + ts.frame_y * (ti.bx * dstress.y);                       // ( 0,  bx,   0) * dstress
//    CudaVec3 dfa = -dfb-dfc;
//
//    atomicAdd(&(df[t.ia].x), -dfa.x);
//    atomicAdd(&(df[t.ia].y), -dfa.y);
//    atomicAdd(&(df[t.ia].z), -dfa.z);
//
//    atomicAdd(&(df[t.ib].x), -dfb.x);
//    atomicAdd(&(df[t.ib].y), -dfb.y);
//    atomicAdd(&(df[t.ib].z), -dfb.z);
//    
//    atomicAdd(&(df[t.ic].x), -dfc.x);
//    atomicAdd(&(df[t.ic].y), -dfc.y);
//    atomicAdd(&(df[t.ic].z), -dfc.z);
//}


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
    mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addForce_kernel"); }
}

void FastTetrahedralCorotationalForceFieldCuda3f_addDForce(unsigned int size, void* df, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu) //, const void* dfdx)
{
    //dim3 threads(BSIZE, 1);
    //dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    //{FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel<float> <<< grid, threads >>> (size, (CudaVec3f*)df, (const CudaVec3f*)dx, kFactor,
    //    (const TriangleState<float>*)triangleState,
    //    (const TriangleInfo<float>*)triangleInfo,
    //    nbTriangles,
    //    (const GPUTriangleInfo*)gpuTriangleInfo,
    //    gamma, mu
    //    ); mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addDForce_kernel"); }
}


#ifdef SOFA_GPU_CUDA_DOUBLE
void FastTetrahedralCorotationalForceFieldCuda3d_addForce(unsigned int size, void* f, const void* x, const void* v,
    unsigned int nbTetrahedra, void* tetrahedronInfo, const void* gpuTetra)
{

    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {

        FastTetrahedralCorotationalForceFieldCudaVec3_addForce_kernel<double> <<< grid, threads >>> (size, (CudaVec3d*)f, (const CudaVec3d*)x, (const CudaVec3d*)v,
            (TriangleState<double>*)triangleState,
            (const TriangleInfo<double>*)triangleInfo,
            nbTriangles,
            (const GPUTriangleInfo*)gpuTriangleInfo,
            gamma, mu
            ); mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addForce_kernel"); }
}

void FastTetrahedralCorotationalForceFieldCuda3d_addDForce(unsigned int size, void* df, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    double gamma, double mu) //, const void* dfdx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {FastTetrahedralCorotationalForceFieldCudaVec3_addDForce_kernel<double> <<< grid, threads >>> (size, (CudaVec3d*)df, (const CudaVec3d*)dx, kFactor,
        (const TriangleState<double>*)triangleState,
        (const TriangleInfo<double>*)triangleInfo,
        nbTriangles,
        (const GPUTriangleInfo*)gpuTriangleInfo,
        gamma, mu
        ); mycudaDebugError("FastTetrahedralCorotationalForceFieldCuda3f_addDForce_kernel"); }
}
#endif

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
