/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
    // I could have written it CudaVec3<float> and CudaVec4<float>, right ?
    float pos[3];
    float rot[4];
};

struct rigidderiv3
{
    // I could have written it with CudaVec3<float> and CudaVec3<float>, right ?
    float3 pos;
    float3 rot;
};

template<>
class CudaRigidCoord3<float> : public rigidcoord3
{
public:
    typedef float Real;
    static __inline__ __device__ __host__ CudaRigidCoord3<float> make(Real x, Real y, Real z, Real rx, Real ry, Real rz, Real rw)
    {
        CudaRigidCoord3<float> r; r.pos[0] = x; r.pos[1] = y; r.pos[2] = z; r.rot[0] = rx; r.rot[1] = ry; r.rot[2] = rz; r.rot[3] = rw; return r;
    }
    static __inline__ __device__ __host__ CudaRigidCoord3<float> make(CudaRigidCoord3<float> v)
    {
        CudaRigidCoord3<float> r; r.pos[0] = v.pos[0]; r.pos[1] = v.pos[1]; r.pos[2] = v.pos[2]; r.rot[0] = v.rot[0]; r.rot[1] = v.rot[1]; r.rot[2] = v.rot[2]; r.rot[3] = v.rot[3]; return r;
    }

};

template<class real>
__device__ void operator+=(CudaRigidCoord3<real>& a, CudaRigidCoord3<real> b)
{
    a.pos[0] += b.pos[0];
    a.pos[1] += b.pos[1];
    a.pos[2] += b.pos[2];

    CudaVec4<real> rotA = CudaVec4<real>::make(a.rot[0], a.rot[1], a.rot[2], a.rot[3]);
    CudaVec4<real> rotB = CudaVec4<real>::make(b.rot[0], b.rot[1], b.rot[2], b.rot[3]);

    rotA = rotB*rotA;
    a.rot[0] = rotA.x;
    a.rot[1] = rotA.y;
    a.rot[2] = rotA.z;
    a.rot[3] = rotA.w;
}

template<class real>
__device__ CudaRigidCoord3<real> operator+(CudaRigidCoord3<real> a, CudaRigidCoord3<real> b)
{
    CudaRigidCoord3<real> sum;
    sum.pos[0] = a.pos[0] + b.pos[0];
    sum.pos[1] = a.pos[1] + b.pos[1];
    sum.pos[2] = a.pos[2] + b.pos[2];

    CudaVec4<real> rotA = CudaVec4<real>::make(a.rot[0], a.rot[1], a.rot[2], a.rot[3]);
    CudaVec4<real> rotB = CudaVec4<real>::make(b.rot[0], b.rot[1], b.rot[2], b.rot[3]);

    rotA = rotB*rotA;
    sum.rot[0] = rotA.x;
    sum.rot[1] = rotA.y;
    sum.rot[2] = rotA.z;
    sum.rot[3] = rotA.w;

    return sum;
}

// template<class real>
// __device__ void operator=(CudaRigidCoord3<real>& a, CudaRigidCoord3<real> b)
// {
//   a.pos[0] = b.pos[0];
//   a.pos[1] = b.pos[1];
//   a.pos[2] = b.pos[2];
//
//   a.rot[0] = b.rot[0];
//   a.rot[1] = b.rot[1];
//   a.rot[2] = b.rot[2];
//   a.rot[3] = b.rot[3];
// }

template<>
class CudaRigidDeriv3<float> : public rigidderiv3
{
public:
    typedef float Real;
    static __inline__ __device__ __host__ CudaRigidDeriv3<float> make(Real x, Real y, Real z, Real rx, Real ry, Real rz)
    {
        CudaRigidDeriv3<float> r; r.pos.x = x; r.pos.y = y; r.pos.z = z; r.rot.x =
            rx; r.rot.y = ry; r.rot.z = rz; return r;
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

template<class real>
__device__ CudaRigidCoord3<real> operator+(CudaRigidCoord3<real> a, CudaRigidDeriv3<real> b)
{
    CudaRigidCoord3<real> x;

    x.pos[0] = a.pos[0] + b.pos.x;
    x.pos[1] = a.pos[1] + b.pos.y;
    x.pos[2] = a.pos[2] + b.pos.z;

    CudaVec4<real> orient = CudaVec4<real>::make(a.rot[0], a.rot[1], a.rot[2], a.rot[3]);
    orient = orient*invnorm(orient);
    CudaVec3<real> vOrient = CudaVec3<real>::make(b.rot.x, b.rot.y, b.rot.z);
    CudaVec4<real> qDot = vectQuatMult(orient, vOrient);
    orient.x += qDot.x*0.5f;
    orient.y += qDot.y*0.5f;
    orient.z += qDot.z*0.5f;
    orient.w += qDot.w*0.5f;
    orient = orient*invnorm(orient);

    x.rot[0] = orient.x;
    x.rot[1] = orient.y;
    x.rot[2] = orient.z;
    x.rot[3] = orient.w;

    return x;
}

template<typename real>
__device__ CudaRigidDeriv3<real> operator*(CudaRigidDeriv3<real> lhs, real rhs)
{
	CudaRigidDeriv3<real> r = lhs;
	
	r.pos.x *= rhs;
	r.pos.y *= rhs;
	r.pos.z *= rhs;
	
	r.rot.x *= rhs;
	r.rot.y *= rhs;
	r.rot.z *= rhs;
	r.rot.w *= rhs;

	return r;
}

template<typename real>
__device__ CudaRigidDeriv3<real> operator*(real lhs, CudaRigidDeriv3<real> rhs)
{
	CudaRigidDeriv3<real> r = rhs;
	
	r.pos.x *= lhs;
	r.pos.y *= lhs;
	r.pos.z *= lhs;
	
	r.rot.x *= lhs;
	r.rot.y *= lhs;
	r.rot.z *= lhs;
	r.rot.w *= lhs;

	return r;
}

#ifdef SOFA_GPU_CUDA_DOUBLE

struct rigidcoord3d
{
    double pos[3];
    double rot[4];
};

struct rigidderiv3d
{
    double3 pos;
    double3 rot;
};

template<>
class CudaRigidCoord3<double> : public rigidcoord3d
{
public:
    typedef double Real;
    static __inline__ __device__ __host__ CudaRigidCoord3<double> make(Real x, Real y, Real z, Real rx, Real ry, Real rz, Real rw)
    {
        CudaRigidCoord3<double> r; r.pos[0] = x; r.pos[1] = y; r.pos[2] = z; r.rot[0] = rx; r.rot[1] = ry; r.rot[2] = rz; r.rot[3] = rw; return r;
    }
    static __inline__ __device__ __host__ CudaRigidCoord3<double> make(CudaRigidCoord3<double> v)
    {
        CudaRigidCoord3<double> r; r.pos[0] = v.pos[0]; r.pos[1] = v.pos[1]; r.pos[2] = v.pos[2]; r.rot[0] = v.rot[0]; r.rot[1] = v.rot[1]; r.rot[2] = v.rot[2]; r.rot[3] = v.rot[3]; return r;
    }

};

template<>
class CudaRigidDeriv3<double> : public rigidderiv3d
{
public:
    typedef double Real;
    static __inline__ __device__ CudaRigidDeriv3<double> make(Real x, Real y, Real z, Real rx, Real ry, Real rz)
    {
        CudaRigidDeriv3<double> r; r.pos.x = x; r.pos.y = y; r.pos.z = z; r.rot.x =
            rx; r.rot.y = ry, r.rot.z = rz; return r;
    }
    static __inline__ __device__ CudaRigidDeriv3<double> make(CudaRigidDeriv3<double> v)
    {
        CudaRigidDeriv3<double> r; r.pos.x = v.pos.x; r.pos.y = v.pos.y; r.pos.z = v.pos.z; r.rot.x = v.rot.x; r.rot.y = v.rot.y; r.rot.z = v.rot.z; return r;
    }
    static __inline__ __device__ CudaRigidDeriv3<double> make(CudaVec3<double> pos, CudaVec3<double> rot)
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
