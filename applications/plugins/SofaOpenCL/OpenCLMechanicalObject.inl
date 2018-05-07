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
#ifndef SOFAOPENCL_OPENCLMECHANICALOBJECT_INL
#define SOFAOPENCL_OPENCLMECHANICALOBJECT_INL

#include "OpenCLMechanicalObject.h"
#include <SofaBaseMechanics/MechanicalObject.inl>
#include <SofaBaseMechanics/MappedObject.inl>
#include <stdio.h>

#define DEBUG_TEXT(t) //printf("   %s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace gpu
{

namespace opencl
{



extern "C"
{
    extern void MechanicalObjectOpenCLVec3f_vAssign(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f_vClear(size_t size,  OpenCLMemoryManager<float>::device_pointer res);
    extern void MechanicalObjectOpenCLVec3f_vMEq(size_t size, _device_pointer res, float f);

    extern void MechanicalObjectOpenCLVec3f_vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f);
    extern void MechanicalObjectOpenCLVec3f_vPEq(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f);
    extern void MechanicalObjectOpenCLVec3f_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b);
    extern void MechanicalObjectOpenCLVec3f_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f);

    extern void MechanicalObjectOpenCLVec3f_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    extern void MechanicalObjectOpenCLVec3f_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2);

    extern void MechanicalObjectOpenCLVec3f_vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, float f11, const _device_pointer b12, float f12, const _device_pointer b13, float f13, const _device_pointer b14, float f14,
            _device_pointer res2, const _device_pointer b21, float f21, const _device_pointer b22, float f22, const _device_pointer b23, float f23, const _device_pointer b24, float f24);
    extern void MechanicalObjectOpenCLVec3f_vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, float f2);
    extern int MechanicalObjectOpenCLVec3f_vDotTmpSize(size_t size);
    extern void MechanicalObjectOpenCLVec3f_vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp);

    extern void MechanicalObjectOpenCLVec3f1_vAssign(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f1_vClear(size_t size, _device_pointer res);
    extern void MechanicalObjectOpenCLVec3f1_vMEq(size_t size, _device_pointer res, float f);
    extern void MechanicalObjectOpenCLVec3f1_vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f);
    extern void MechanicalObjectOpenCLVec3f1_vPEq(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f1_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f);
    extern void MechanicalObjectOpenCLVec3f1_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b);
    extern void MechanicalObjectOpenCLVec3f1_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f);
    extern void MechanicalObjectOpenCLVec3f1_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    extern void MechanicalObjectOpenCLVec3f1_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2);
    extern void MechanicalObjectOpenCLVec3f1_vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, float f11, const _device_pointer b12, float f12, const _device_pointer b13, float f13, const _device_pointer b14, float f14,
            _device_pointer res2, const _device_pointer b21, float f21, const _device_pointer b22, float f22, const _device_pointer b23, float f23, const _device_pointer b24, float f24);
    extern void MechanicalObjectOpenCLVec3f1_vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, float f2);
    extern int MechanicalObjectOpenCLVec3f1_vDotTmpSize(size_t size);
    extern void MechanicalObjectOpenCLVec3f1_vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp);



    extern void MechanicalObjectOpenCLVec3d_vAssign(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3d_vClear(size_t size,  OpenCLMemoryManager<double>::device_pointer res);
    extern void MechanicalObjectOpenCLVec3d_vMEq(size_t size, _device_pointer res, double f);
    extern void MechanicalObjectOpenCLVec3d_vEqBF(size_t size, _device_pointer res, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d_vPEq(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3d_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b);
    extern void MechanicalObjectOpenCLVec3d_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
    extern void MechanicalObjectOpenCLVec3d_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer b2, double f2);
    extern void MechanicalObjectOpenCLVec3d_vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, double f11, const _device_pointer b12, double f12, const _device_pointer b13, double f13, const _device_pointer b14, double f14,
            _device_pointer res2, const _device_pointer b21, double f21, const _device_pointer b22, double f22, const _device_pointer b23, double f23, const _device_pointer b24, double f24);
    extern void MechanicalObjectOpenCLVec3d_vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, double f2);
    extern int MechanicalObjectOpenCLVec3d_vDotTmpSize(size_t size);
    extern void MechanicalObjectOpenCLVec3d_vDot(size_t size, double* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, double* cputmp);
    extern void MechanicalObjectOpenCLVec3d1_vAssign(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3d1_vClear(size_t size, _device_pointer res);
    extern void MechanicalObjectOpenCLVec3d1_vMEq(size_t size, _device_pointer res, double f);
    extern void MechanicalObjectOpenCLVec3d1_vEqBF(size_t size, _device_pointer res, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d1_vPEq(size_t size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3d1_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d1_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b);
    extern void MechanicalObjectOpenCLVec3d1_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f);
    extern void MechanicalObjectOpenCLVec3d1_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
    extern void MechanicalObjectOpenCLVec3d1_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer b2, double f2);
    extern void MechanicalObjectOpenCLVec3d1_vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, double f11, const _device_pointer b12, double f12, const _device_pointer b13, double f13, const _device_pointer b14, double f14,
            _device_pointer res2, const _device_pointer b21, double f21, const _device_pointer b22, double f22, const _device_pointer b23, double f23, const _device_pointer b24, double f24);
    extern void MechanicalObjectOpenCLVec3d1_vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, double f2);
    extern int MechanicalObjectOpenCLVec3d1_vDotTmpSize(size_t size);
    extern void MechanicalObjectOpenCLVec3d1_vDot(size_t size, double* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, double* cputmp);



} // extern "C"


template<>
class OpenCLKernelsMechanicalObject<OpenCLVec3fTypes>
{
public:


    static void vAssign(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3f_vAssign(size, res, a); }
    static void vClear(size_t size,  _device_pointer res)
    {   MechanicalObjectOpenCLVec3f_vClear(size, res); }
    static void vMEq(size_t size, _device_pointer res, float f)
    {   MechanicalObjectOpenCLVec3f_vMEq(size, res, f); }
    static void vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f_vEqBF(size, res, b, f); }
    static void vPEq(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3f_vPEq(size, res, a); }
    static void vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f_vPEqBF(size, res, b, f); }
    static void vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
    {   MechanicalObjectOpenCLVec3f_vAdd(size, res, a, b); }
    static void vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f_vOp(size, res, a, b, f); }
    static void vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectOpenCLVec3f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2)
    {   MechanicalObjectOpenCLVec3f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, float f11, const _device_pointer b12, float f12, const _device_pointer b13, float f13, const _device_pointer b14, float f14,
            _device_pointer res2, const _device_pointer b21, float f21, const _device_pointer b22, float f22, const _device_pointer b23, float f23, const _device_pointer b24, float f24)
    {
        MechanicalObjectOpenCLVec3f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, float f2)
    {   MechanicalObjectOpenCLVec3f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(size_t size)
    {   return MechanicalObjectOpenCLVec3f_vDotTmpSize(size); }
    static void vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp)
    {   MechanicalObjectOpenCLVec3f_vDot(size, res, a, b, tmp, cputmp); }
};

template<>
class OpenCLKernelsMechanicalObject<OpenCLVec3f1Types>
{
public:
    static void vAssign(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3f1_vAssign(size, res, a); }
    static void vClear(size_t size, _device_pointer res)
    {   MechanicalObjectOpenCLVec3f1_vClear(size, res); }
    static void vMEq(size_t size, _device_pointer res, float f)
    {   MechanicalObjectOpenCLVec3f1_vMEq(size, res, f); }
    static void vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f1_vEqBF(size, res, b, f); }
    static void vPEq(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3f1_vPEq(size, res, a); }
    static void vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f1_vPEqBF(size, res, b, f); }
    static void vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
    {   MechanicalObjectOpenCLVec3f1_vAdd(size, res, a, b); }
    static void vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f)
    {   MechanicalObjectOpenCLVec3f1_vOp(size, res, a, b, f); }
    static void vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectOpenCLVec3f1_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2)
    {   MechanicalObjectOpenCLVec3f1_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, float f11, const _device_pointer b12, float f12, const _device_pointer b13, float f13, const _device_pointer b14, float f14,
            _device_pointer res2, const _device_pointer b21, float f21, const _device_pointer b22, float f22, const _device_pointer b23, float f23, const _device_pointer b24, float f24)
    {
        MechanicalObjectOpenCLVec3f1_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, float f2)
    {   MechanicalObjectOpenCLVec3f1_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(size_t size)
    {   return MechanicalObjectOpenCLVec3f1_vDotTmpSize(size); }
    static void vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp)
    {   MechanicalObjectOpenCLVec3f1_vDot(size, res, a, b, tmp, cputmp); }
};


template<>
class OpenCLKernelsMechanicalObject<OpenCLVec3dTypes>
{
public:
    static void vAssign(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3d_vAssign(size, res, a); }
    static void vClear(size_t size,  OpenCLMemoryManager<double>::device_pointer res)
    {   MechanicalObjectOpenCLVec3d_vClear(size, res); }
    static void vMEq(size_t size, _device_pointer res, double f)
    {   MechanicalObjectOpenCLVec3d_vMEq(size, res, f); }
    static void vEqBF(size_t size, _device_pointer res, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d_vEqBF(size, res, b, f); }
    static void vPEq(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3d_vPEq(size, res, a); }
    static void vPEqBF(size_t size, _device_pointer res, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d_vPEqBF(size, res, b, f); }
    static void vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
    {   MechanicalObjectOpenCLVec3d_vAdd(size, res, a, b); }
    static void vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d_vOp(size, res, a, b, f); }
    static void vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
    {   MechanicalObjectOpenCLVec3d_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer b2, double f2)
    {   MechanicalObjectOpenCLVec3d_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, double f11, const _device_pointer b12, double f12, const _device_pointer b13, double f13, const _device_pointer b14, double f14,
            _device_pointer res2, const _device_pointer b21, double f21, const _device_pointer b22, double f22, const _device_pointer b23, double f23, const _device_pointer b24, double f24)
    {
        MechanicalObjectOpenCLVec3d_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, double f2)
    {   MechanicalObjectOpenCLVec3d_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(size_t size)
    {   return MechanicalObjectOpenCLVec3d_vDotTmpSize(size); }
    static void vDot(size_t size, double* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, double* cputmp)
    {   MechanicalObjectOpenCLVec3d_vDot(size, res, a, b, tmp, cputmp); }
};

template<>
class OpenCLKernelsMechanicalObject<OpenCLVec3d1Types>
{
public:
    static void vAssign(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3d1_vAssign(size, res, a); }
    static void vClear(size_t size, _device_pointer res)
    {   MechanicalObjectOpenCLVec3d1_vClear(size, res); }
    static void vMEq(size_t size, _device_pointer res, double f)
    {   MechanicalObjectOpenCLVec3d1_vMEq(size, res, f); }
    static void vEqBF(size_t size, _device_pointer res, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d1_vEqBF(size, res, b, f); }
    static void vPEq(size_t size, _device_pointer res, const _device_pointer a)
    {   MechanicalObjectOpenCLVec3d1_vPEq(size, res, a); }
    static void vPEqBF(size_t size, _device_pointer res, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d1_vPEqBF(size, res, b, f); }
    static void vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
    {   MechanicalObjectOpenCLVec3d1_vAdd(size, res, a, b); }
    static void vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f)
    {   MechanicalObjectOpenCLVec3d1_vOp(size, res, a, b, f); }
    static void vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
    {   MechanicalObjectOpenCLVec3d1_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer b2, double f2)
    {   MechanicalObjectOpenCLVec3d1_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, double f11, const _device_pointer b12, double f12, const _device_pointer b13, double f13, const _device_pointer b14, double f14,
            _device_pointer res2, const _device_pointer b21, double f21, const _device_pointer b22, double f22, const _device_pointer b23, double f23, const _device_pointer b24, double f24)
    {
        MechanicalObjectOpenCLVec3d1_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, double f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, double f2)
    {   MechanicalObjectOpenCLVec3d1_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(size_t size)
    {   return MechanicalObjectOpenCLVec3d1_vDotTmpSize(size); }
    static void vDot(size_t size, double* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, double* cputmp)
    {   MechanicalObjectOpenCLVec3d1_vDot(size, res, a, b, tmp, cputmp); }
};




} // namespace opencl

} // namespace gpu





namespace component
{

namespace container
{

using namespace gpu::opencl;

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::accumulateForce(Main* m)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::accumulateForce ");
    if (!m->externalForces.getValue().empty())
    {
        //std::cout << "ADD: external forces, size = "<< m->externalForces.getValue().size() << std::endl;
        Kernels::vAssign(m->externalForces.getValue().size(), m->f.beginEdit()->deviceWrite(), m->externalForces.getValue().deviceRead());
        m->f.endEdit();
    }
    //else std::cout << "NO external forces" << std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vAlloc(Main* m, VecId v)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::vAlloc ");
    if (v.type == sofa::core::V_COORD && v.index >= core::VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = m->getVecCoord(v.index);
        vec->recreate(m->d_size.getValue());
    }
    else if (v.type == sofa::core::V_DERIV && v.index >= core::VecDerivId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = m->getVecDeriv(v.index);
        vec->recreate(m->d_size.getValue());
    }
    else
    {
        std::cerr << "Invalid alloc operation ("<<v<<")\n";
        return;
    }
    //vOp(v); // clear vector
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, SReal f)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vOp ");
    if(v.isNull())
    {
        // ERROR
        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
        return;
    }
    //std::cout << "> vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
    if (a.isNull())
    {
        if (b.isNull())
        {
            // v = 0
            if (v.type == sofa::core::V_COORD)
            {
                Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                VecCoord* vv = d_vv->beginEdit();
                vv->recreate(m->d_size.getValue());
                Kernels::vClear(vv->size(), vv->deviceWrite());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                VecDeriv* vv = d_vv->beginEdit();
                vv->recreate(m->d_size.getValue());
                Kernels::vClear(vv->size(), vv->deviceWrite());
                d_vv->endEdit();
            }
        }
        else
        {
            if (b.type != v.type)
            {
                // ERROR
                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                return;
            }
            if (v == b)
            {
                // v *= f
                if (v.type == sofa::core::V_COORD)
                {
                    Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                    VecCoord* vv = d_vv->beginEdit();
                    Kernels::vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                    VecDeriv* vv = d_vv->beginEdit();
                    Kernels::vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                    d_vv->endEdit();
                }
            }
            else
            {
                // v = b*f
                if (v.type == sofa::core::V_COORD)
                {
                    Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                    const Data<VecCoord>* d_vb = m->read((core::ConstVecCoordId)b);
                    VecCoord* vv = d_vv->beginEdit();
                    const VecCoord* vb = &d_vb->getValue();
                    vv->recreate(vb->size());
                    Kernels::vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                    const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                    VecDeriv* vv = d_vv->beginEdit();
                    const VecDeriv* vb = &d_vb->getValue();
                    vv->recreate(vb->size());
                    Kernels::vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                    d_vv->endEdit();
                }
            }
        }
    }
    else
    {
        if (a.type != v.type)
        {
            // ERROR
            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
            return;
        }
        if (b.isNull())
        {
            // v = a
            if (v.type == sofa::core::V_COORD)
            {
                Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                const Data<VecCoord>* d_va = m->read((core::ConstVecCoordId)a);
                VecCoord* vv = d_vv->beginEdit();
                const VecCoord* va = &d_va->getValue();
                vv->recreate(va->size());
                Kernels::vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                const Data<VecDeriv>* d_va = m->read((core::ConstVecDerivId)a);
                VecDeriv* vv = d_vv->beginEdit();
                const VecDeriv* va = &d_va->getValue();
                vv->recreate(va->size());
                Kernels::vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
                d_vv->endEdit();
            }
        }
        else
        {
            if (v == a)
            {
                if (f==1.0)
                {
                    // v += b
                    if (v.type == sofa::core::V_COORD)
                    {
                        Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                        VecCoord* vv = d_vv->beginEdit();
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((core::ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            if (vb->size() > vv->size())
                                vv->resize(vb->size());
                            if (vb->size()>0)
                                Kernels::vPEq(vb->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            if (vb->size() > vv->size())
                                vv->resize(vb->size());
                            if (vb->size()>0)
                                Kernels::vPEq(vb->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                        const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                        VecDeriv* vv = d_vv->beginEdit();
                        const VecDeriv* vb = &d_vb->getValue();
                        if (vb->size() > vv->size())
                            vv->resize(vb->size());
                        if (vb->size() > 0)
                            Kernels::vPEq(vb->size(), vv->deviceWrite(), vb->deviceRead());
                        d_vv->endEdit();
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v += b*f
                    if (v.type == sofa::core::V_COORD)
                    {
                        Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                        VecCoord* vv = d_vv->beginEdit();
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((core::ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                        const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                        VecDeriv* vv = d_vv->beginEdit();
                        const VecDeriv* vb = &d_vb->getValue();
                        vv->resize(vb->size());
                        Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        d_vv->endEdit();
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
            else
            {
                if (f==1.0)
                {
                    // v = a+b
                    if (v.type == sofa::core::V_COORD)
                    {
                        Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                        const Data<VecCoord>* d_va = m->read((core::ConstVecCoordId)a);
                        VecCoord* vv = d_vv->beginEdit();
                        const VecCoord* va = &d_va->getValue();
                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((core::ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                        VecDeriv* vv = d_vv->beginEdit();
                        const Data<VecDeriv>* d_va = m->read((core::ConstVecDerivId)a);
                        const VecDeriv* va = &d_va->getValue();
                        const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                        const VecDeriv* vb = &d_vb->getValue();
                        vv->recreate(va->size());
                        Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        d_vv->endEdit();
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+b*f
                    if (v.type == sofa::core::V_COORD)
                    {
                        Data<VecCoord>* d_vv = m->write((core::VecCoordId)v);
                        const Data<VecCoord>* d_va = m->read((core::ConstVecCoordId)a);
                        VecCoord* vv = d_vv->beginEdit();
                        const VecCoord* va = &d_va->getValue();

                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((core::ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }

                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                        VecDeriv* vv = d_vv->beginEdit();
                        const Data<VecDeriv>* d_va = m->read((core::ConstVecDerivId)a);
                        const VecDeriv* va = &d_va->getValue();
                        const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                        const VecDeriv* vb = &d_vb->getValue();
                        vv->recreate(va->size());
                        Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        d_vv->endEdit();
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
        }
    }
    //std::cout << "< vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
    DEBUG_TEXT("~MechanicalObjectInternalData::vOp ");
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vMultiOp ");

    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2
        && ops[0].second.size() == 2
        && ops[0].first.getId(m) == ops[0].second[0].first.getId(m)
        && ops[0].first.getId(m).type == sofa::core::V_DERIV
        && ops[0].second[1].first.getId(m).type == sofa::core::V_DERIV
        && ops[1].second.size() == 2
        && ops[1].first.getId(m) == ops[1].second[0].first.getId(m)
        && ops[0].first.getId(m) == ops[1].second[1].first.getId(m)
        && ops[1].first.getId(m).type == sofa::core::V_COORD)
    {
        const Data<VecDeriv>* d_va = m->read(core::ConstVecDerivId(ops[0].second[1].first.getId(m)));
        const VecDeriv* va = &d_va->getValue();
        Data<VecDeriv>* d_vv = m->write(core::VecDerivId(ops[0].first.getId(m)));
        VecDeriv* vv = d_vv->beginEdit();
        Data<VecCoord>* d_vx = m->write(core::VecCoordId(ops[1].first.getId(m)));
        VecDeriv* vx = d_vx->beginEdit();
        const unsigned int n = vx->size();
        const double f_v_v = ops[0].second[0].second;
        const double f_v_a = ops[0].second[1].second;
        const double f_x_x = ops[1].second[0].second;
        const double f_x_v = ops[1].second[1].second;
        Kernels::vIntegrate(n, va->deviceRead(), vv->deviceWrite(), vx->deviceWrite(), (Real)f_v_v, (Real)f_v_a, (Real)f_x_x, (Real)f_x_v);
        d_vv->endEdit();
        d_vx->endEdit();
    }
    // optimize common CG step: x += a*p, q -= a*v
    else if (ops.size() == 2 && ops[0].second.size() == 2
            && ops[0].first.getId(m) == ops[0].second[0].first.getId(m)
            && ops[0].second[0].second == 1.0
            && ops[0].first.getId(m).type == sofa::core::V_DERIV
            && ops[0].second[1].first.getId(m).type == sofa::core::V_DERIV
            && ops[1].second.size() == 2
            && ops[1].first.getId(m) == ops[1].second[0].first.getId(m)
            && ops[1].second[0].second == 1.0
            && ops[1].first.getId(m).type == sofa::core::V_DERIV
            && ops[1].second[1].first.getId(m).type == sofa::core::V_DERIV)
    {
        const Data<VecDeriv>* d_vv1 = m->read(core::ConstVecDerivId(ops[0].second[1].first.getId(m)));
        const VecDeriv* vv1 = &d_vv1->getValue();
        const Data<VecDeriv>* d_vv2 = m->read(core::ConstVecDerivId(ops[1].second[1].first.getId(m)));
        const VecDeriv* vv2 = &d_vv2->getValue();

        Data<VecDeriv>* d_vres1 = m->write(core::VecDerivId(ops[0].first.getId(m)));
        VecDeriv* vres1 = d_vres1->beginEdit();
        Data<VecDeriv>* d_vres2 = m->write(core::VecDerivId(ops[1].first.getId(m)));
        VecDeriv* vres2 = d_vres2->beginEdit();

        const unsigned int n = vres1->size();
        const double f1 = ops[0].second[1].second;
        const double f2 = ops[1].second[1].second;
        Kernels::vPEqBF2(n, vres1->deviceWrite(), vv1->deviceRead(), f1, vres2->deviceWrite(), vv2->deviceRead(), f2);

        d_vres1->endEdit();
        d_vres2->endEdit();
    }
    // optimize a pair of generic vOps
    else if (ops.size()==2
            && ops[0].second.size()==2
            && ops[0].second[0].second == 1.0
            && ops[1].second.size()==2
            && ops[1].second[0].second == 1.0)
    {
        const unsigned int n = m->getSize();

        _device_pointer w0Ptr, r0Ptr0, r0Ptr1;
        _device_pointer w1Ptr, r1Ptr0, r1Ptr1;

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        r0Ptr0 = (ops[0].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[0].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[0].first.getId(m)))->getValue().deviceRead();
        r0Ptr1 = (ops[0].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[1].first.getId(m)))->getValue().deviceRead();
        w1Ptr  = (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();
        r1Ptr0 = (ops[1].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[0].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[0].first.getId(m)))->getValue().deviceRead();
        r1Ptr1 = (ops[1].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[1].first.getId(m)))->getValue().deviceRead();

        Kernels::vOp2(n, w0Ptr, r0Ptr0, r0Ptr1,	ops[0].second[1].second, w1Ptr, r1Ptr0, r1Ptr1, ops[1].second[1].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[0].first.getId(m)))->endEdit();
        (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[1].first.getId(m)))->endEdit();
    }
    // optimize a pair of 4-way accumulations (such as at the end of RK4)
    else if (ops.size()==2
            && ops[0].second.size()==5
            && ops[0].second[0].first.getId(m) == ops[0].first.getId(m)
            && ops[0].second[0].second == 1.0
            && ops[1].second.size()==5
            && ops[1].second[0].first.getId(m) == ops[1].first.getId(m)
            && ops[1].second[0].second == 1.0)
    {
        const unsigned int n = m->getSize();

        _device_pointer w0Ptr, r0Ptr[4];
        _device_pointer w1Ptr, r1Ptr[4];

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        w1Ptr  = (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();

        for(unsigned int i=0 ; i < 4 ; i++)
        {
            r0Ptr[i] = (ops[0].second[i+1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[i+1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[i+1].first.getId(m)))->getValue().deviceRead();
            r1Ptr[i] = (ops[1].second[i+1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[i+1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[i+1].first.getId(m)))->getValue().deviceRead();;

        }
        Kernels::vPEq4BF2(n, w0Ptr, r0Ptr[0], ops[0].second[1].second, r0Ptr[1], ops[0].second[2].second, r0Ptr[2], ops[0].second[3].second, r0Ptr[3], ops[0].second[4].second,
                w1Ptr, r1Ptr[0], ops[1].second[1].second, r1Ptr[1], ops[1].second[2].second, r1Ptr[2], ops[1].second[3].second, r1Ptr[3], ops[1].second[4].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[0].first.getId(m)))->endEdit();
        (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[1].first.getId(m)))->endEdit();
    }
    else // no optimization for now for other cases
    {
        std::cout << "OPENCL: unoptimized vMultiOp:"<<std::endl;
        for (unsigned int i=0; i<ops.size(); ++i)
        {
            std::cout << ops[i].first << " =";
            if (ops[i].second.empty())
                std::cout << "0";
            else
                for (unsigned int j=0; j<ops[i].second.size(); ++j)
                {
                    if (j) std::cout << " + ";
                    std::cout << ops[i].second[j].first << "*" << ops[i].second[j].second;
                }
            std::cout << std::endl;
        }
        {
            using namespace sofa::core::behavior;
            m->BaseMechanicalState::vMultiOp(params, ops);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
SReal MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vDot(Main* m, ConstVecId a, ConstVecId b)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vDot ");
    Real r = 0.0f;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        const VecCoord* va = &m->read(core::ConstVecCoordId(a))->getValue();
        const VecCoord* vb = &m->read(core::ConstVecCoordId(b))->getValue();
        int tmpsize = Kernels::vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), gpu::opencl::_device_pointer(), NULL);
        }
        else
        {
            m->data.tmpdot.recreate(tmpsize);
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), m->data.tmpdot.deviceWrite(), (Real*)(&(m->data.tmpdot.getCached(0))));
        }
    }
    else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
    {
        const VecDeriv* va = &m->read(core::ConstVecDerivId(a))->getValue();
        const VecDeriv* vb = &m->read(core::ConstVecDerivId(b))->getValue();
        int tmpsize = Kernels::vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), gpu::opencl::_device_pointer(), NULL);
        }
        else
        {
            m->data.tmpdot.recreate(tmpsize);
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), m->data.tmpdot.deviceWrite(), (Real*)(&(m->data.tmpdot.getCached(0))));
        }
#ifndef NDEBUG
        // Check the result
        //Real r2 = 0.0f;
        //for (unsigned int i=0; i<va->size(); i++)
        //	r2 += (*va)[i] * (*vb)[i];
        //std::cout << "OPENCL vDot: GPU="<<r<<"  CPU="<<r2<<" relative error="<<(fabsf(r2)>0.000001?fabsf(r-r2)/fabsf(r2):fabsf(r-r2))<<"\n";
#endif
    }
    else
    {
        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
    }

    DEBUG_TEXT("~MechanicalObjectInternalData::vDot ");
    return r;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::resetForce(Main* m)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::resetForce ");
    Data<VecDeriv>* d_f = m->write(core::VecDerivId::force());
    VecDeriv& f = *d_f->beginEdit();
    if (f.size() > 0)
        Kernels::vClear(f.size(), f.deviceWrite());
    d_f->endEdit();
}


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLMechanicalObject_ImplMethods(T) \
template<> void MechanicalObject< T >::accumulateForce(const core::ExecParams* params, core::VecDerivId fid) \
{ if( fid==core::VecDerivId::force() ) data.accumulateForce(this); else core::behavior::BaseMechanicalState::accumulateForce(params,fid); } \
template<> void MechanicalObject< T >::vOp(const core::ExecParams* /* params */ /* PARAMS FIRST */, core::VecId v, core::ConstVecId a, core::ConstVecId b, SReal f) \
{ data.vOp(this, v, a, b, f); }		\
template<> void MechanicalObject< T >::vMultiOp(const core::ExecParams* params /* PARAMS FIRST */, const VMultiOp& ops) \
{ data.vMultiOp(this, params, ops); }                                    \
template<> SReal MechanicalObject< T >::vDot(const core::ExecParams* /* params */ /* PARAMS FIRST */, core::ConstVecId a, core::ConstVecId b) \
{ return data.vDot(this, a, b); }				    \
template<> void MechanicalObject< T >::resetForce(const core::ExecParams* params, core::VecDerivId fid) \
{ if( fid==core::VecDerivId::force() ) data.resetForce(this); else core::behavior::BaseMechanicalState::resetForce(params,fid); }

OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3fTypes)
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3f1Types)
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3dTypes)
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3d1Types)

#undef OpenCLMechanicalObject_ImplMethods


} // namespace container

} // namespace component

} // namespace sofa

#undef DEBUG_TEXT

#endif
