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
#include "cuda.h"


#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

struct GPUTriangleInfo
{
    int ia, ib, ic;
};

template <typename real>
struct TriangleInfo
{
    //Index ia, ib, ic;
    real bx, cx, cy, ss_factor;

    //Transformation init_frame; // Mat<2,3,Real>
    CudaVec3<real> init_frame_x;
    CudaVec3<real> init_frame_y;
};

template <typename real>
struct TriangleState
{
    //Transformation frame; // Mat<2,3,Real>
    CudaVec3<real> frame_x;
    CudaVec3<real> frame_y;
    CudaVec3<real> stress;
};


//////////////////////
// GPU-side methods //
//////////////////////

template <typename real>
__global__ void TriangularFEMForceFieldOptimCudaVec3_addForce_kernel(int size, CudaVec3<real>* f, const CudaVec3<real>* x, const CudaVec3<real>* v,
    TriangleState<real>* triState, const TriangleInfo<real>* triInfo,
    unsigned int nbTriangles,
    const GPUTriangleInfo* gpuTriangleInfo,
    real gamma, real mu)
{
    using CudaVec3 = CudaVec3<real>;
    const int triangleID = (blockIdx.x * BSIZE) + threadIdx.x;

    const GPUTriangleInfo& t = gpuTriangleInfo[triangleID];
    const TriangleInfo<real>& ti = triInfo[triangleID];
    TriangleState<real>& ts = triState[triangleID];

    const CudaVec3 a = x[t.ia];
    const CudaVec3 ab = x[t.ib] - a;
    const CudaVec3 ac = x[t.ic] - a;
            
    // compute locale frame inside the triangle frame: [ab] x [ac]
    CudaVec3 frame_x = ab;
    CudaVec3 n = cross(ab, ac);
    CudaVec3 frame_y = cross(n, ab);
    frame_x *= invnorm(frame_x);
    frame_y *= invnorm(frame_y);

    // save triangle frame computed at this dt
    ts.frame_x = frame_x; 
    ts.frame_y = frame_y;

    // compute local displacement (dby = 0 as ab == frame_x)
    float dbx = ti.bx - dot(frame_x, ab);
    float dcx = ti.cx - dot(frame_x, ac);
    float dcy = ti.cy - dot(frame_y, ac);
        

    CudaVec3 strain = CudaVec3::make (
        ti.cy * dbx,                // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
        ti.bx * dcy,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
        ti.bx * dcx - ti.cx * dbx); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

    const float gammaXY = (strain.x + strain.y) * gamma;

    CudaVec3 stress = CudaVec3::make (
        mu*strain.x + gammaXY,     // (gamma+mu, gamma   ,    0) * strain
        mu*strain.y + gammaXY,     // (gamma   , gamma+mu,    0) * strain
        (real)(0.5)*mu*strain.z);  // (       0,        0, mu/2) * strain

    ts.stress = stress;

    stress *= ti.ss_factor;
    CudaVec3 fb = frame_x * (ti.cy * stress.x - ti.cx * stress.z)  // (cy,   0, -cx) * stress
            + frame_y * (ti.cy * stress.z - ti.cx * stress.y);     // ( 0, -cx,  cy) * stress
    CudaVec3 fc = frame_x * (ti.bx * stress.z)                     // ( 0,   0,  bx) * stress
            + frame_y * (ti.bx * stress.y);                        // ( 0,  bx,   0) * stress
    CudaVec3 fa = -fb-fc;

    // f[triangle[0]] += fa;
    atomicAdd(&(f[t.ia].x), fa.x);
    atomicAdd(&(f[t.ia].y), fa.y);
    atomicAdd(&(f[t.ia].z), fa.z);

    // f[triangle[1]] += fb;
    atomicAdd(&(f[t.ib].x), fb.x);
    atomicAdd(&(f[t.ib].y), fb.y);
    atomicAdd(&(f[t.ib].z), fb.z);
    
    // f[triangle[2]] += fc;
    atomicAdd(&(f[t.ic].x), fc.x);
    atomicAdd(&(f[t.ic].y), fc.y);
    atomicAdd(&(f[t.ic].z), fc.z);
}


template <typename real>
__global__ void TriangularFEMForceFieldOptimCudaVec3_addDForce_kernel(int size, CudaVec3<real>* df, const CudaVec3<real>* dx, real kFactor,
    const TriangleState<real>* triState, const TriangleInfo<real>* triInfo,
    unsigned int nbTriangles,
    const GPUTriangleInfo* gpuTriangleInfo,
    real gamma, real mu
)
{
    using CudaVec3 = CudaVec3<real>;
    const int triangleID = (blockIdx.x * BSIZE) + threadIdx.x;
    
    const GPUTriangleInfo& t = gpuTriangleInfo[triangleID];
    const TriangleInfo<real>& ti = triInfo[triangleID];
    const TriangleState<real>& ts = triState[triangleID];

    const CudaVec3 da  = dx[t.ia];
    const CudaVec3 dab = dx[t.ib]-da;
    const CudaVec3 dac = dx[t.ic]-da;
    const real dbx = dot(ts.frame_x, dab);
    const real dby = dot(ts.frame_y, dab);
    const real dcx = dot(ts.frame_x, dac);
    const real dcy = dot(ts.frame_y, dac);

    const CudaVec3 dstrain = CudaVec3::make (
        ti.cy  * dbx,                             // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
        ti.bx * dcy - ti.cx * dby,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
        ti.bx * dcx - ti.cx * dbx + ti.cy * dby); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

    const real gammaXY = (dstrain.x + dstrain.y) * gamma;

    CudaVec3 dstress = CudaVec3::make (
        mu*dstrain.x + gammaXY,    // (gamma+mu, gamma   ,    0) * dstrain
        mu*dstrain.y + gammaXY,    // (gamma   , gamma+mu,    0) * dstrain
        (float)(0.5)*mu*dstrain.z); // (       0,        0, mu/2) * dstrain

    dstress *= ti.ss_factor * kFactor;
    CudaVec3 dfb = ts.frame_x * (ti.cy * dstress.x - ti.cx * dstress.z)  // (cy,   0, -cx) * dstress
            + ts.frame_y * (ti.cy * dstress.z - ti.cx * dstress.y);   // ( 0, -cx,  cy) * dstress
    CudaVec3 dfc = ts.frame_x * (ti.bx * dstress.z)                      // ( 0,   0,  bx) * dstress
            + ts.frame_y * (ti.bx * dstress.y);                       // ( 0,  bx,   0) * dstress
    CudaVec3 dfa = -dfb-dfc;

    atomicAdd(&(df[t.ia].x), -dfa.x);
    atomicAdd(&(df[t.ia].y), -dfa.y);
    atomicAdd(&(df[t.ia].z), -dfa.z);

    atomicAdd(&(df[t.ib].x), -dfb.x);
    atomicAdd(&(df[t.ib].y), -dfb.y);
    atomicAdd(&(df[t.ib].z), -dfb.z);
    
    atomicAdd(&(df[t.ic].x), -dfc.x);
    atomicAdd(&(df[t.ic].y), -dfc.y);
    atomicAdd(&(df[t.ic].z), -dfc.z);
}


//////////////////////
// CPU-side methods //
//////////////////////

extern "C"
{

void TriangularFEMForceFieldOptimCuda3f_addForce(unsigned int size, void* f, const void* x, const void* v,
    void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu)
{

    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {

        TriangularFEMForceFieldOptimCudaVec3_addForce_kernel<float> <<< grid, threads >>> (size, (CudaVec3f*)f, (const CudaVec3f*)x, (const CudaVec3f*)v,
            (TriangleState<float>*)triangleState,
            (const TriangleInfo<float>*)triangleInfo,
            nbTriangles,
            (const GPUTriangleInfo*)gpuTriangleInfo,
            gamma, mu
            ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addForce_kernel"); }
}

void TriangularFEMForceFieldOptimCuda3f_addDForce(unsigned int size, void* df, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu) //, const void* dfdx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {TriangularFEMForceFieldOptimCudaVec3_addDForce_kernel<float> <<< grid, threads >>> (size, (CudaVec3f*)df, (const CudaVec3f*)dx, kFactor,
        (const TriangleState<float>*)triangleState,
        (const TriangleInfo<float>*)triangleInfo,
        nbTriangles,
        (const GPUTriangleInfo*)gpuTriangleInfo,
        gamma, mu
        ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addDForce_kernel"); }
}


#ifdef SOFA_GPU_CUDA_DOUBLE
void TriangularFEMForceFieldOptimCuda3d_addForce(unsigned int size, void* f, const void* x, const void* v,
    void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    double gamma, double mu)
{

    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {

        TriangularFEMForceFieldOptimCudaVec3_addForce_kernel<double> <<< grid, threads >>> (size, (CudaVec3d*)f, (const CudaVec3d*)x, (const CudaVec3d*)v,
            (TriangleState<double>*)triangleState,
            (const TriangleInfo<double>*)triangleInfo,
            nbTriangles,
            (const GPUTriangleInfo*)gpuTriangleInfo,
            gamma, mu
            ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addForce_kernel"); }
}

void TriangularFEMForceFieldOptimCuda3d_addDForce(unsigned int size, void* df, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    double gamma, double mu) //, const void* dfdx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((nbTriangles + BSIZE - 1) / BSIZE, 1);
    {TriangularFEMForceFieldOptimCudaVec3_addDForce_kernel<double> <<< grid, threads >>> (size, (CudaVec3d*)df, (const CudaVec3d*)dx, kFactor,
        (const TriangleState<double>*)triangleState,
        (const TriangleInfo<double>*)triangleInfo,
        nbTriangles,
        (const GPUTriangleInfo*)gpuTriangleInfo,
        gamma, mu
        ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addDForce_kernel"); }
}
#endif

} // extern "C"

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
