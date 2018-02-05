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
#ifndef CUDAMATH_INL
#define CUDAMATH_INL

#include "CudaMath.h"
#include <SofaCUDA/config.h>

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
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec3<double> operator*(CudaVec3<double> a, double b)
{
    return CudaVec3<double>::make(__dmul_rn(a.x, b), __dmul_rn(a.y, b), __dmul_rn(a.z, b));
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> operator/(CudaVec3<float> a, float b)
{
    return CudaVec3<float>::make(__fdiv_rn(a.x, b), __fdiv_rn(a.y, b), __fdiv_rn(a.z, b));
}
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE_PRECISE
template<>
__device__ CudaVec3<double> operator/(CudaVec3<double> a, double b)
{
    return CudaVec3<double>::make(__ddiv_rn(a.x, b), __ddiv_rn(a.y, b), __ddiv_rn(a.z, b));
}
#endif // SOFA_GPU_CUDA_PRECISE


#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ void operator*=(CudaVec3<float>& a, float b)
{
    a.x = __fmul_rn(a.x, b);
    a.y = __fmul_rn(a.y, b);
    a.z = __fmul_rn(a.z, b);
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ void operator*=(CudaVec3<double>& a, double b)
{
    a.x = __dmul_rn(a.x, b);
    a.y = __dmul_rn(a.y, b);
    a.z = __dmul_rn(a.z, b);
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> operator*(float a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a, b.x), __fmul_rn(a, b.y), __fmul_rn(a, b.z));
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec3<double> operator*(double a, CudaVec3<double> b)
{
    return CudaVec3<double>::make(__dmul_rn(a, b.x), __dmul_rn(a, b.y), __dmul_rn(a, b.z));
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> mul(CudaVec3<float> a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y), __fmul_rn(a.z, b.z));
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec3<double> mul(CudaVec3<double> a, CudaVec3<double> b)
{
    return CudaVec3<double>::make(__dmul_rn(a.x, b.x), __dmul_rn(a.y, b.y), __dmul_rn(a.z, b.z));
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float dot(CudaVec3<float> a, CudaVec3<float> b)
{
    return __fmul_rn(a.x, b.x) + __fmul_rn(a.y, b.y) + __fmul_rn(a.z, b.z);
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ double dot(CudaVec3<double> a, CudaVec3<double> b)
{
    return __dmul_rn(a.x, b.x) + __dmul_rn(a.y, b.y) + __dmul_rn(a.z, b.z);
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec3<float> cross(CudaVec3<float> a, CudaVec3<float> b)
{
    return CudaVec3<float>::make(__fmul_rn(a.y, b.z)-__fmul_rn(a.z, b.y), __fmul_rn(a.z, b.x)-__fmul_rn(a.x, b.z), __fmul_rn(a.x, b.y)-__fmul_rn(a.y, b.x));
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec3<double> cross(CudaVec3<double> a, CudaVec3<double> b)
{
    return CudaVec3<double>::make(__dmul_rn(a.y, b.z)-__dmul_rn(a.z, b.y), __dmul_rn(a.z, b.x)-__dmul_rn(a.x, b.z), __dmul_rn(a.x, b.y)-__dmul_rn(a.y, b.x));
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm2(CudaVec3<float> a)
{
    return __fmul_rn(a.x, a.x)+__fmul_rn(a.y, a.y)+__fmul_rn(a.z, a.z);
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ double norm2(CudaVec3<double> a)
{
    return __dmul_rn(a.x, a.x)+__dmul_rn(a.y, a.y)+__dmul_rn(a.z, a.z);
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm(CudaVec3<float> a)
{
    return __fsqrt_rn(norm2(a));
}
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_DOUBLE_PRECISE
template<>
__device__ double norm(CudaVec3<double> a)
{
    return __dsqrt_rn(norm2(a));
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
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_DOUBLE_PRECISE
template<>
__device__ double invnorm(CudaVec3<double> a)
{
    if (norm2(a) > 0.0)
        return __ddiv_rn(1.0f, __dsqrt_rn(norm2(a)));
    return 0.0;
}
#endif

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float norm2(CudaVec4<float> a)
{
    return __fmul_rn(a.x, a.x)+__fmul_rn(a.y, a.y)+__fmul_rn(a.z, a.z)+__fmul_rn(a.w, a.w);
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ double norm2(CudaVec4<double> a)
{
    return __dmul_rn(a.x, a.x)+__dmul_rn(a.y, a.y)+__dmul_rn(a.z, a.z)+__dmul_rn(a.w, a.w);
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ float invnorm(CudaVec4<float> a)
{
    if (norm2(a) > 0.0)
        return __fdiv_rn(1.0f, __fsqrt_rn(norm2(a)));
    return 0.0;
}
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_DOUBLE_PRECISE
template<>
__device__ double invnorm(CudaVec4<double> a)
{
    if (norm2(a) > 0.0)
        return __ddiv_rn(1.0f, __dsqrt_rn(norm2(a)));
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
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_DOUBLE_PRECISE
template<>
__device__ CudaVec4<double> inv(CudaVec4<double> a)
{
    float norm = norm2(a);
    if (norm > 0)
        return CudaVec4<double>::make(__ddiv_rn(-a.y, norm), __ddiv_rn(-a.y, norm), __ddiv_rn(-a.z, norm), __ddiv_rn(a.w, norm));
    return CudaVec4<double>::make(0.0, 0.0, 0.0, 0.0);
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
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec4<double> operator*(CudaVec4<double> a, CudaVec4<double> b)
{
    CudaVec4<double> quat;
    quat.w = __dmul_rn(a.w, b.w) - (__dmul_rn(a.x, b.x) + __dmul_rn(a.y, b.y) + __dmul_rn(a.z, b.z));
    quat.x = __dmul_rn(a.w, b.x) + __dmul_rn(a.x, b.w) + __dmul_rn(a.y, b.z) - __dmul_rn(a.z, b.y);
    quat.y = __dmul_rn(a.w, b.y) + __dmul_rn(a.y, b.w) + __dmul_rn(a.z, b.x) - __dmul_rn(a.x, b.z);
    quat.z = __dmul_rn(a.w, b.z) + __dmul_rn(a.z, b.w) + __dmul_rn(a.x, b.y) - __dmul_rn(a.y, b.x);

    return quat;
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

#ifdef SOFA_GPU_CUDA_PRECISE
template<>
__device__ CudaVec4<float> operator*(CudaVec4<float> a, float f)
{
    return CudaVec4<float>::make(__fmul_rn(a.x, f), __fmul_rn(a.y, f), __fmul_rn(a.z, f), __fmul_rn(a.w, f));
}
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec4<double> operator*(CudaVec4<double> a, double f)
{
    return CudaVec4<double>::make(__dmul_rn(a.x, f), __dmul_rn(a.y, f), __dmul_rn(a.z, f), __dmul_rn(a.w, f));
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

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
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec4<double> createQuaterFromEuler(CudaVec3<double> v)
{
    CudaVec4<double> quat;
    double a0 = v.x;
    a0 /= (double)2.0;
    double a1 = v.y;
    a1 /= (double)2.0;
    double a2 = v.z;
    a2 /= (double)2.0;
    quat.w = __dmul_rn(__dmul_rn(cos(a0),cos(a1)),cos(a2)) + __dmul_rn(__dmul_rn(sin(a0),sin(a1)),sin(a2));
    quat.x = __dmul_rn(__dmul_rn(sin(a0),cos(a1)),cos(a2)) - __dmul_rn(__dmul_rn(cos(a0),sin(a1)),sin(a2));
    quat.y = __dmul_rn(__dmul_rn(cos(a0),sin(a1)),cos(a2)) + __dmul_rn(__dmul_rn(sin(a0),cos(a1)),sin(a2));
    quat.z = __dmul_rn(__dmul_rn(cos(a0),cos(a1)),sin(a2)) - __dmul_rn(__dmul_rn(sin(a0),sin(a1)),cos(a2));
    return quat;
}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE

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
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
__device__ CudaVec4<double> vectQuatMult(CudaVec4<double> a, const CudaVec3<double> vect)
{
    CudaVec4<double>	ret;

    ret.w = (double) (-(__dmul_rn(vect.x, a.x) + __dmul_rn(vect.y, a.y) + __dmul_rn(vect.z, a.z)));
    ret.x = (double) (__dmul_rn(vect.x, a.w) + __dmul_rn(vect.y, a.z) - __dmul_rn(vect.z, a.y));
    ret.y = (double) (__dmul_rn(vect.y, a.w) + __dmul_rn(vect.z, a.x) - __dmul_rn(vect.x, a.z));
    ret.z = (double) (__dmul_rn(vect.z, a.w) + __dmul_rn(vect.x, a.y) - __dmul_rn(vect.y, a.x));

    return ret;

}
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_GPU_CUDA_PRECISE





#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
