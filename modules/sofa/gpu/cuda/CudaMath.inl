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
#ifndef CUDAMATH_INL
#define CUDAMATH_INL

#include "CudaMath.h"

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> operator*(CudaVec3<float> a, float b)
{
    return CudaVec3<float>::make(__fmul_rn(a.x, b), __fmul_rn(a.y, b), __fmul_rn(a.z, b));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> operator/(CudaVec3<float> a, float b)
{
    return CudaVec3<float>::make(__fdiv_rn(a.x, b), __fdiv_rn(a.y, b), __fdiv_rn(a.z, b));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ void operator*=(CudaVec3<float>& a, float b)
{
    a.x = __fmul_rn(a.x, b);
    a.y = __fmul_rn(a.y, b);
    a.z = __fmul_rn(a.z, b);
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> operator*(float a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a, b.x), __fmul_rn(a, b.y), __fmul_rn(a, b.z));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> mul(CudaVec3<float> a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y), __fmul_rn(a.z, b.z));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float dot(CudaVec3<float> a, CudaVec3<float> b)
{
    return __fmul_rn(a.x, b.x) + __fmul_rn(a.y, b.y) + __fmul_rn(a.z, b.z);
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> cross(CudaVec3<float> a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a.y, b.z)-__fmul_rn(a.z, b.y), __fmul_rn(a.z, b.x)-__fmul_rn(a.x, b.z), __fmul_rn(a.x, b.y)-__fmul_rn(a.y, b.x));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm2(CudaVec3<float> a)
{
    return __fmul_rn(a.x, a.x)+__fmul_rn(a.y, a.y)+__fmul_rn(a.z, a.z);
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm(CudaVec3<float> a)
{
    return __fsqrt_rn(norm2(a));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float invnorm(CudaVec3<float> a)
{
    if (norm2(a) > 0.0)
        return __fdiv_rn(1.0f, __fsqrt_rn(norm2(a)));
    return 0.0;
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm2(CudaVec4<float> a)
{
    return __fmul_rn(a.x, a.x)+__fmul_rn(a.y, a.y)+__fmul_rn(a.z, a.z)+__fmul_rn(a.w, a.w);
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float invnorm(CudaVec4<float> a)
{
    if (norm2(a) > 0.0)
        return __fdiv_rn(1.0f, __fsqrt_rn(norm2(a)));
    return 0.0;
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> inv(CudaVec4<float> a)
{
    float norm = norm2(a);
    if (norm > 0)
        return CudaVec4<float>::make(__fdiv_rn(-a.y, norm), __fdiv_rn(-a.y, norm), __fdiv_rn(-a.z, norm), __fdiv_rn(a.w, norm));
    return CudaVec4<float>::make(0.0, 0.0, 0.0, 0.0);
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> operator*(CudaVec4<float> a, CudaVec4<float> b)
{
    CudaVec4<float> quat;
    quat.w = __fmul_rn(a.w, b.w) - (__fmul_rn(a.x, b.x) + __fmul_rn(a.y, b.y) + __fmul_rn(a.z, b.z));
    quat.x = __fmul_rn(a.w, b.x) + __fmul_rn(a.x, b.w) + __fmul_rn(a.y, b.z) - __fmul_rn(a.z, b.y);
    quat.y = __fmul_rn(a.w, b.y) + __fmul_rn(a.y, b.w) + __fmul_rn(a.z, b.x) - __fmul_rn(a.x, b.z);
    quat.z = __fmul_rn(a.w, b.z) + __fmul_rn(a.z, b.w) + __fmul_rn(a.x, b.y) - __fmul_rn(a.y, b.x);

    return quat;
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> operator*(CudaVec4<float> a, float f)
{
    return CudaVec4<float>::make(__fmul_rn(a.x, f), __fmul_rn(a.y, f), __fmul_rn(a.z, f), __fmul_rn(a.w, f));
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> createQuaterFromEuler(CudaVec3<float> v)
{
    CudaVec4<float> quat;
    float a0 = v.x;
    a0 /= (float)2.0;
    float a1 = v.y;
    a1 /= (float)2.0;
    float a2 = v.z;
    a2 /= (float)2.0;
    quat.w = __fmul_rn(__fmul_rn(cos(a0),cos(a1)),cos(a2)) + __fmul_rn(__fmul_rn(sin(a0),sin(a1)),sin(a2));
    quat.x = __fmul_rn(__fmul_rn(sin(a0),cos(a1)),cos(a2)) - __fmul_rn(__fmul_rn(cos(a0),sin(a1)),sin(a2));
    quat.y = __fmul_rn(__fmul_rn(cos(a0),sin(a1)),cos(a2)) + __fmul_rn(__fmul_rn(sin(a0),cos(a1)),sin(a2));
    quat.z = __fmul_rn(__fmul_rn(cos(a0),cos(a1)),sin(a2)) - __fmul_rn(__fmul_rn(sin(a0),sin(a1)),cos(a2));
    return quat;
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> vectQuatMult(CudaVec4<float> a, const CudaVec3<float> vect)
{
    CudaVec4<float>	ret;

    ret.w = (float) (-(__fmul_rn(vect.x, a.x) + __fmul_rn(vect.y, a.y) + __fmul_rn(vect.z, a.z)));
    ret.x = (float) (__fmul_rn(vect.x, a.w) + __fmul_rn(vect.y, a.z) - __fmul_rn(vect.z, a.y));
    ret.y = (float) (__fmul_rn(vect.y, a.w) + __fmul_rn(vect.z, a.x) - __fmul_rn(vect.x, a.z));
    ret.z = (float) (__fmul_rn(vect.z, a.w) + __fmul_rn(vect.x, a.y) - __fmul_rn(vect.y, a.x));

    return ret;

}
#endif





#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
