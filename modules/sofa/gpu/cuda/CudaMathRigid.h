/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef CUDAMATHRIGID_H
#define CUDAMATHRIGID_H

#include <cuda_runtime.h>
#include <cuda.h>

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

template<class real>
class CudaRigidCoord3;

template<class real>
class CudaRigidDeriv3;

struct rigidcoord3
{
    float pos[3];
    float rot[4];
};

struct rigidderiv3
{
    float3 pos;
    float3 rot;
};

template<>
class CudaRigidCoord3<float> : public rigidcoord3
{

};

template<>
class CudaRigidDeriv3<float> : public rigidderiv3
{
public:
    typedef float Real;
    static __inline__ __device__ __host__ CudaRigidDeriv3<float> make(Real x, Real y, Real z, Real rx, Real ry, Real rz)
    {
        CudaRigidDeriv3<float> r; r.pos.x = x; r.pos.y = y; r.pos.z = z; r.rot.x =
            rx; r.rot.y = ry, r.rot.z = rz; return r;
    }
    static __inline__ __device__ __host__ CudaRigidDeriv3<float> make(CudaRigidDeriv3<float> v)
    {
        CudaRigidDeriv3<float> r; r.pos.x = v.pos.x; r.pos.y = v.pos.y; r.pos.z = v.pos.z; r.rot.x = v.rot.x; r.rot.y = v.rot.y; r.rot.z = v.rot.z; return r;
    }
    static __inline__ __device__ __host__ CudaRigidDeriv3<float> make(CudaVec3<float> pos, CudaVec3<float> rot)
    {
        CudaRigidDeriv3<float> r; r.pos.x = pos.x; r.pos.y = pos.y; r.pos.z = pos.z; r.rot.x = rot.x; r.rot.y = rot.y; r.rot.z = rot.z; return r;
    }
};

template<class real>
__device__ void operator+=(CudaRigidDeriv3<real>& a, CudaRigidDeriv3<real> b)
{
    a.pos.x += b.pos.x;
    a.pos.y += b.pos.y;
    a.pos.z += b.pos.z;
    a.rot.x += b.rot.x;
    a.rot.y += b.rot.y;
    a.rot.z += b.rot.z;
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaRigidCoord3<double> : public rigidcoord3
{

};

template<>
class CudaRigidDeriv3<double> : public rigidderiv3
{
public:
    typedef double Real;
    static __inline__ __device__ __host__ CudaRigidDeriv3<double> make(Real x, Real y, Real z, Real rx, Real ry, Real rz)
    {
        CudaRigidDeriv3<double> r; r.pos.x = x; r.pos.y = y; r.pos.z = z; r.rot.x =
            rx; r.rot.y = ry, r.rot.z = rz; return r;
    }
    static __inline__ __device__ __host__ CudaRigidDeriv3<double> make(CudaRigidDeriv3<double> v)
    {
        CudaRigidDeriv3<double> r; r.pos.x = v.pos.x; r.pos.y = v.pos.y; r.pos.z = v.pos.z; r.rot.x = v.rot.x; r.rot.y = v.rot.y; r.rot.z = v.rot.z; return r;
    }
    static __inline__ __device__ __host__ CudaRigidDeriv3<double> make(CudaVec3<double> pos, CudaVec3<double> rot)
    {
        CudaRigidDeriv3<double> r; r.pos.x = pos.x; r.pos.y = pos.y; r.pos.z = pos.z; r.rot.x = rot.x; r.rot.y = rot.y; r.rot.z = rot.z; return r;
    }
};

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
#endif
