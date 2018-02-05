/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ < 200
#if defined(_MSC_VER)
#pragma warning __CUDA_ARCH__ is too low for atomics
#else
#warning __CUDA_ARCH__ is too low for atomics
#endif
#endif

#if defined(__cplusplus) && CUDA_VERSION < 2000
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

struct TriangleInfo
{
    //Index ia, ib, ic;
    float bx, cx, cy, ss_factor;
    //Transformation init_frame; // Mat<2,3,Real>
    CudaVec3<float> init_frame_x;
    CudaVec3<float> init_frame_y;
};

struct TriangleState
{
    //Transformation frame; // Mat<2,3,Real>
    CudaVec3<float> frame_x;
    CudaVec3<float> frame_y;
    CudaVec3<float> stress;
};

extern "C"
{
void TriangularFEMForceFieldOptimCuda3f_addForce(unsigned int size, void* f, const void* x, const void* v,
    void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu);
    void TriangularFEMForceFieldOptimCuda3f_addDForce(unsigned int size, void* f, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu); //, const void* dfdx);

}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__device__ real operator*(CudaVec3<real> a, CudaVec3<real> b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}
/*
template<class real>
__device__ CudaVec3<real> operator*(CudaVec3<real> a, real b)
{
    return CudaVec3<real>::make(a.x*b, a.y*b, a.z*b);
}
*/
__global__ void TriangularFEMForceFieldOptimCuda3f_addForce_kernel(int size, CudaVec3<float>* f, const CudaVec3<float>* x, const CudaVec3<float>* v,
    TriangleState* triState, const TriangleInfo* triInfo,
    unsigned int nbTriangles,
    const GPUTriangleInfo* gpuTriangleInfo,
    float gamma, float mu
)
{
    int index0 = (blockIdx.x*BSIZE);
    int index = threadIdx.x;
    int i = index0+index;
    
        GPUTriangleInfo t = gpuTriangleInfo[i];
        const TriangleInfo& ti = triInfo[i];
        TriangleState& ts = triState[i];
        CudaVec3<float> a  = x[t.ia];
        CudaVec3<float> ab = x[t.ib]-a;
        CudaVec3<float> ac = x[t.ic]-a;
        //computeTriangleRotation(ts.frame, ab, ac);
        CudaVec3<float> frame_x = ab;
        CudaVec3<float> n = cross(ab,ac);
        CudaVec3<float> frame_y = cross(n,ab);
        frame_x *= invnorm(frame_x);
        frame_y *= invnorm(frame_y);
        ts.frame_x = frame_x;
        ts.frame_y = frame_y;

        float dbx = ti.bx - frame_x*ab;
        // float dby = 0
        float dcx = ti.cx - frame_x*ac;
        float dcy = ti.cy - frame_y*ac;
        //sout << "Elem" << i << ": D= 0 0  " << dbx << " 0  " << dcx << " " << dcy << sendl;

        CudaVec3<float> strain = CudaVec3<float>::make (
            ti.cy * dbx,                // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

        float gammaXY = (strain.x+strain.y)*gamma;

        CudaVec3<float> stress = CudaVec3<float>::make (
            mu*strain.x + gammaXY,    // (gamma+mu, gamma   ,    0) * strain
            mu*strain.y + gammaXY,    // (gamma   , gamma+mu,    0) * strain
            (float)(0.5)*mu*strain.z); // (       0,        0, mu/2) * strain

        ts.stress = stress;

        stress *= ti.ss_factor;
        //sout << "Elem" << i << ": F= " << -(ti.cy * stress[0] - ti.cx * stress[2] + ti.bx * stress[2]) << " " << -(ti.cy * stress[2] - ti.cx * stress[1] + ti.bx * stress[1]) << "  " << (ti.cy * stress[0] - ti.cx * stress[2]) << " " << (ti.cy * stress[2] - ti.cx * stress[1]) << "  " << (ti.bx * stress[2]) << " " << (ti.bx * stress[1]) << sendl;
        CudaVec3<float> fb = frame_x * (ti.cy * stress.x - ti.cx * stress.z)  // (cy,   0, -cx) * stress
                + frame_y * (ti.cy * stress.z - ti.cx * stress.y); // ( 0, -cx,  cy) * stress
        CudaVec3<float> fc = frame_x * (ti.bx * stress.z)                      // ( 0,   0,  bx) * stress
                + frame_y * (ti.bx * stress.y);                     // ( 0,  bx,   0) * stress
        CudaVec3<float> fa = -fb-fc;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ < 200
        f[t.ia] += fa;
        f[t.ib] += fb;
        f[t.ic] += fc;
#else
        atomicAdd(&(f[t.ia].x), fa.x);
        atomicAdd(&(f[t.ia].y), fa.y);
        atomicAdd(&(f[t.ia].z), fa.z);
        atomicAdd(&(f[t.ib].x), fb.x);
        atomicAdd(&(f[t.ib].y), fb.y);
        atomicAdd(&(f[t.ib].z), fb.z);
        atomicAdd(&(f[t.ic].x), fc.x);
        atomicAdd(&(f[t.ic].y), fc.y);
        atomicAdd(&(f[t.ic].z), fc.z);
#endif
}

typedef CudaVec3<float> Coord;
typedef CudaVec3<float> Deriv;
typedef float Real;

__global__ void TriangularFEMForceFieldOptimCuda3f_addDForce_kernel(int size, CudaVec3<float>* df, const CudaVec3<float>* dx, float kFactor,
    const TriangleState* triState, const TriangleInfo* triInfo,
    unsigned int nbTriangles,
    const GPUTriangleInfo* gpuTriangleInfo,
    float gamma, float mu
)
{
    int index0 = (blockIdx.x*BSIZE);
    int index = threadIdx.x;
    int i = index0+index;
    
    GPUTriangleInfo t = gpuTriangleInfo[i];
    const TriangleInfo& ti = triInfo[i];
    const TriangleState& ts = triState[i];

        Deriv da  = dx[t.ia];
        Deriv dab = dx[t.ib]-da;
        Deriv dac = dx[t.ic]-da;
        Real dbx = ts.frame_x*dab;
        Real dby = ts.frame_y*dab;
        Real dcx = ts.frame_x*dac;
        Real dcy = ts.frame_y*dac;

        CudaVec3<float> dstrain = CudaVec3<float>::make (
            ti.cy  * dbx,                             // ( cy,   0,  0,  0) * (dbx, dby, dcx, dcy)
            ti.bx * dcy - ti.cx * dby,                // (  0, -cx,  0, bx) * (dbx, dby, dcx, dcy)
            ti.bx * dcx - ti.cx * dbx + ti.cy * dby); // (-cx,  cy, bx,  0) * (dbx, dby, dcx, dcy)

        Real gammaXY = (dstrain.x+dstrain.y)*gamma;

        CudaVec3<float> dstress = CudaVec3<float>::make (
            mu*dstrain.x + gammaXY,    // (gamma+mu, gamma   ,    0) * dstrain
            mu*dstrain.y + gammaXY,    // (gamma   , gamma+mu,    0) * dstrain
            (Real)(0.5)*mu*dstrain.z); // (       0,        0, mu/2) * dstrain

        dstress *= ti.ss_factor * kFactor;
        Deriv dfb = ts.frame_x * (ti.cy * dstress.x - ti.cx * dstress.z)  // (cy,   0, -cx) * dstress
                + ts.frame_y * (ti.cy * dstress.z - ti.cx * dstress.y); // ( 0, -cx,  cy) * dstress
        Deriv dfc = ts.frame_x * (ti.bx * dstress.z)                       // ( 0,   0,  bx) * dstress
                + ts.frame_y * (ti.bx * dstress.y);                      // ( 0,  bx,   0) * dstress
        Deriv dfa = -dfb-dfc;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ < 200
        df[t.ia] -= dfa;
        df[t.ib] -= dfb;
        df[t.ic] -= dfc;
#else
        atomicAdd(&(df[t.ia].x), -dfa.x);
        atomicAdd(&(df[t.ia].y), -dfa.y);
        atomicAdd(&(df[t.ia].z), -dfa.z);
        atomicAdd(&(df[t.ib].x), -dfb.x);
        atomicAdd(&(df[t.ib].y), -dfb.y);
        atomicAdd(&(df[t.ib].z), -dfb.z);
        atomicAdd(&(df[t.ic].x), -dfc.x);
        atomicAdd(&(df[t.ic].y), -dfc.y);
        atomicAdd(&(df[t.ic].z), -dfc.z);
#endif
}

//////////////////////
// CPU-side methods //
//////////////////////

void TriangularFEMForceFieldOptimCuda3f_addForce(unsigned int size, void* f, const void* x, const void* v,
    void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTriangles+BSIZE-1)/BSIZE,1);
    {TriangularFEMForceFieldOptimCuda3f_addForce_kernel<<< grid, threads >>>(size, (CudaVec3<float>*)f, (const CudaVec3<float>*)x, (const CudaVec3<float>*)v,
                                                                                                    (TriangleState*) triangleState,
                                                                                                    (const TriangleInfo*) triangleInfo,
                                                                                                    nbTriangles,
                                                                                                    (const GPUTriangleInfo*) gpuTriangleInfo,
                                                                                                    gamma, mu
        ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addForce_kernel");}
}

void TriangularFEMForceFieldOptimCuda3f_addDForce(unsigned int size, void* df, const void* dx, float kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    float gamma, float mu) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTriangles+BSIZE-1)/BSIZE,1);
    {TriangularFEMForceFieldOptimCuda3f_addDForce_kernel<<< grid, threads>>>(size, (CudaVec3<float>*)df, (const CudaVec3<float>*)dx, kFactor,
                                                                                                    (const TriangleState*) triangleState,
                                                                                                    (const TriangleInfo*) triangleInfo,
                                                                                                    nbTriangles,
                                                                                                    (const GPUTriangleInfo*) gpuTriangleInfo,
                                                                                                    gamma, mu
        ); mycudaDebugError("TriangularFEMForceFieldOptimCuda3f_addDForce_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
