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
#ifndef SOFA_GPU_OPENCL_OPENCLMECHANICALOBJECT_INL
#define SOFA_GPU_OPENCL_OPENCLMECHANICALOBJECT_INL

#include "OpenCLMechanicalObject.h"
#include <sofa/component/container/MechanicalObject.inl>
#include <sofa/component/container/MappedObject.inl>
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

    /*
    struct VDotOp
    {
    	const void* a;
    	const void* b;
    	int size;
    };
    */

    struct VDotOp
    {
        /*const*/ _device_pointer a;
        /*const*/ _device_pointer b;
        int size;
    };


    extern int MultiMechanicalObjectOpenCLVec3f_vDotTmpSize(size_t n, VDotOp* ops);
    extern void MultiMechanicalObjectOpenCLVec3f_vDot(size_t n, VDotOp* ops, double* results, _device_pointer tmp, float* cputmp);


    /*
    struct VOpF
    {
    	void* res;
    	const void* a;
    	const void* b;
    	double f;
    	int size;
    };
    */

    /*modif
    struct VOpF
    {
    	sofa::gpu::opencl::_device_pointer res;
    	//const
    	sofa::gpu::opencl::_device_pointer a;
    	//const
    	 sofa::gpu::opencl::_device_pointer b;
    	float f;
    	int size;
    };
    */

    struct VOpF
    {
        _device_pointer res;
        //const
        _device_pointer a;
        //const
        _device_pointer b;
        float f;
        int size;
    };

    /*
    struct VOpD
    {
    	void* res;
    	const void* a;
    	const void* b;
    	double f;
    	int size;
    };*/

    struct VOpD
    {
        _device_pointer res;
        /*const*/ _device_pointer a;
        /*const*/ _device_pointer b;
        float f;
        int size;
    };

    extern void MultiMechanicalObjectOpenCLVec3f_vOp(size_t n, VOpF* ops);

    /*
    struct VClearOp
    {
    	void* res;
    	int size;
    };
    */

    struct VClearOp
    {
        _device_pointer res;
        int size;
    };

    extern void MultiMechanicalObjectOpenCLVec3f_vClear(size_t n, VClearOp* ops);

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
    static bool supportMultiVDot() { return myopenclMultiOpMax>0; }
    static int multiVDotTmpSize(size_t n, VDotOp* ops)
    {   return MultiMechanicalObjectOpenCLVec3f_vDotTmpSize(n, ops); }
    static void multiVDot(size_t n, VDotOp* ops, double* results, _device_pointer tmp, float* cputmp)
    {   MultiMechanicalObjectOpenCLVec3f_vDot(n, ops, results, tmp, cputmp); }
    typedef VOpF VOp;
    static bool supportMultiVOp() { return myopenclMultiOpMax>0; }
    static void multiVOp(size_t n, VOp* ops)
    {   MultiMechanicalObjectOpenCLVec3f_vOp(n, ops); }
    static bool supportMultiVClear() { return myopenclMultiOpMax>0; }
    static void multiVClear(size_t n, VClearOp* ops)
    {   MultiMechanicalObjectOpenCLVec3f_vClear(n, ops); }
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
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(size_t, VDotOp*)
    {   return 0; }
    static void multiVDot(size_t, VDotOp*, double*, _device_pointer, float*)
    {}
    typedef VOpF VOp;
    static bool supportMultiVOp() { return false /*myopenclMultiOpMax>0*/; }
    static void multiVOp(size_t /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectOpenCLVec3f1_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(size_t, VClearOp*)
    {}
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
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(size_t, VDotOp*)
    {   return 0; }
    static void multiVDot(size_t, VDotOp*, double*, _device_pointer, double*)
    {}
    typedef VOpD VOp;
    static bool supportMultiVOp() { return false /*myopenclMultiOpMax>0*/; }
    static void multiVOp(size_t /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectOpenCLVec3d_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(size_t, VClearOp*)
    {}
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
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(size_t, VDotOp*)
    {   return 0; }
    static void multiVDot(size_t, VDotOp*, double*, _device_pointer, double*)
    {}
    typedef VOpD VOp;
    static bool supportMultiVOp() { return false /*myopenclMultiOpMax>0*/; }
    static void multiVOp(size_t /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectOpenCLVec3d1_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(unsigned int, VClearOp*)
    {}
};




} // namespace opencl

} // namespace gpu





namespace component
{

namespace container
{

using namespace gpu::opencl;

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::accumulateForce(Main* m, bool prefetch)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::accumulateForce ");
    if (prefetch) return;
    if (!m->externalForces.getValue().empty())
    {
        //std::cout << "ADD: external forces, size = "<< m->externalForces.getValue().size() << std::endl;
        Kernels::vAssign(m->externalForces.getValue().size(), m->f.beginEdit()->deviceWrite(), m->externalForces.getValue().deviceRead());
        m->f.endEdit();
    }
    //else std::cout << "NO external forces" << std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::addDxToCollisionModel(Main* m, bool prefetch)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::addDxToCollisionModel ");
    if (prefetch) return;
    Kernels::vAdd(m->xfree.getValue().size(), m->x.beginEdit()->deviceWrite(), m->xfree.getValue().deviceRead(), m->dx.getValue().deviceRead());
    m->x.endEdit();
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vAlloc(Main* m, VecId v)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::vAlloc ");
    if (v.type == sofa::core::V_COORD && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = m->getVecCoord(v.index);
        vec->recreate(m->vsize);
    }
    else if (v.type == sofa::core::V_DERIV && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = m->getVecDeriv(v.index);
        vec->recreate(m->vsize);
    }
    else
    {
        std::cerr << "Invalid alloc operation ("<<v<<")\n";
        return;
    }
    //vOp(v); // clear vector
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vOp(Main* m, VecId v, VecId a, VecId b, double f, bool prefetch)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vOp ");

    if (prefetch)
    {
        if (!Kernels::supportMultiVOp()) return; // no kernel available for combining multiple operations
        if (v.isNull() || a.isNull() || b.isNull()) return; // ignore invalid or simple operations
        VOp op;
        op.v = v;
        op.a = a;
        op.b = b;
        op.f = f;
        op.size = (a.type == sofa::core::V_COORD) ? m->getVecCoord(a.index)->size() : m->getVecDeriv(a.index)->size();
        m->data.preVOp.push_back(op);
        m->data.preVOp.id = m->data.preVOp.objects().size();
        m->data.preVOp.objects().push_back(m);
        DEBUG_TEXT("~MechanicalObjectInternalData::vOp ");
        return;
    }
    else if (m->data.preVOp.id >= 0)
    {
        helper::vector<Main*>& objects = m->data.preVOp.objects();
        if (!objects.empty())
        {
            if (objects.size() == 1 && m->data.preVOp.size() == 1)
            {
                // only one operation -> use regular kernel
                m->data.preVOp.id = -1;
                m->data.preVOp.clear();
            }
            else
            {
                int nops = 0;
                for (unsigned int i=0; i<objects.size(); ++i)
                    nops += objects[i]->data.preVOp.size();
                helper::vector< typename Kernels::VOp > ops(nops);
                nops = 0;
                for (unsigned int i=0; i<objects.size(); ++i)
                {
                    Main* o = objects[i];
                    helper::vector<VOp>& oops = o->data.preVOp;
                    for (unsigned int j=0; j<oops.size(); ++j)
                    {

                        ops[nops].res = (oops[j].v.type == sofa::core::V_COORD) ? o->getVecCoord(v.index)->deviceWrite() : o->getVecDeriv(v.index)->deviceWrite();
                        ops[nops].a   = (oops[j].a.type == sofa::core::V_COORD) ? o->getVecCoord(a.index)->deviceRead() : o->getVecDeriv(a.index)->deviceRead();
                        ops[nops].b   = (oops[j].b.type == sofa::core::V_COORD) ? o->getVecCoord(b.index)->deviceRead() : o->getVecDeriv(b.index)->deviceRead();
                        ops[nops].f   = (Real)oops[j].f;
                        ops[nops].size = oops[j].size;
                        ++nops;
                    }
                }
                Kernels::multiVOp(nops, &(ops[0]));
            }
            objects.clear();
        }
        if (m->data.preVOp.id != -1) // prefetching was done
        {
            m->data.preVOp.resize(m->data.preVOp.size()-1);
            if (m->data.preVOp.empty())
                m->data.preVOp.id = -1;
            DEBUG_TEXT("~MechanicalObjectInternalData::vOp " );
            return;
        }
    }
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
                VecCoord* vv = m->getVecCoord(v.index);
                vv->recreate(m->vsize);
                Kernels::vClear(vv->size(), vv->deviceWrite());
            }
            else
            {
                VecDeriv* vv = m->getVecDeriv(v.index);
                vv->recreate(m->vsize);
                Kernels::vClear(vv->size(), vv->deviceWrite());
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
                    VecCoord* vv = m->getVecCoord(v.index);
                    Kernels::vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = m->getVecDeriv(v.index);
                    Kernels::vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                }
            }
            else
            {
                // v = b*f
                if (v.type == sofa::core::V_COORD)
                {
                    VecCoord* vv = m->getVecCoord(v.index);
                    VecCoord* vb = m->getVecCoord(b.index);
                    vv->recreate(vb->size());
                    Kernels::vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                }
                else
                {
                    VecDeriv* vv = m->getVecDeriv(v.index);
                    VecDeriv* vb = m->getVecDeriv(b.index);
                    vv->recreate(vb->size());
                    Kernels::vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
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
                VecCoord* vv = m->getVecCoord(v.index);
                VecCoord* va = m->getVecCoord(a.index);
                vv->recreate(va->size());
                Kernels::vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
            }
            else
            {
                VecDeriv* vv = m->getVecDeriv(v.index);
                VecDeriv* va = m->getVecDeriv(a.index);
                vv->recreate(va->size());
                Kernels::vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
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
                        VecCoord* vv = m->getVecCoord(v.index);
                        if (b.type == sofa::core::V_COORD)
                        {
                            VecCoord* vb = m->getVecCoord(b.index);
                            vv->resize(vb->size());
                            Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = m->getVecDeriv(b.index);
                            vv->resize(vb->size());
                            Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        VecDeriv* vv = m->getVecDeriv(v.index);
                        VecDeriv* vb = m->getVecDeriv(b.index);
                        vv->resize(vb->size());
                        Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
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
                        VecCoord* vv = m->getVecCoord(v.index);
                        if (b.type == sofa::core::V_COORD)
                        {
                            VecCoord* vb = m->getVecCoord(b.index);
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = m->getVecDeriv(b.index);
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        VecDeriv* vv = m->getVecDeriv(v.index);
                        VecDeriv* vb = m->getVecDeriv(b.index);
                        vv->resize(vb->size());
                        Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
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
                        VecCoord* vv = m->getVecCoord(v.index);
                        VecCoord* va = m->getVecCoord(a.index);
                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            VecCoord* vb = m->getVecCoord(b.index);
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            VecDeriv* vb = m->getVecDeriv(b.index);
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        VecDeriv* vv = m->getVecDeriv(v.index);
                        VecDeriv* va = m->getVecDeriv(a.index);
                        VecDeriv* vb = m->getVecDeriv(b.index);
                        vv->recreate(va->size());
                        Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
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
                        VecCoord* vv = m->getVecCoord(v.index);
                        VecCoord* va = m->getVecCoord(a.index);
                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            VecCoord* vb = m->getVecCoord(b.index);
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            VecDeriv* vb = m->getVecDeriv(b.index);
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        VecDeriv* vv = m->getVecDeriv(v.index);
                        VecDeriv* va = m->getVecDeriv(a.index);
                        VecDeriv* vb = m->getVecDeriv(b.index);
                        vv->recreate(va->size());
                        Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
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
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vMultiOp(Main* m, const VMultiOp& ops, bool prefetch)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vMultiOp ");

    if (prefetch) return;
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].first.type == sofa::core::V_DERIV && ops[0].second[1].first.type == sofa::core::V_DERIV
        && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[0].first == ops[1].second[1].first && ops[1].first.type == sofa::core::V_COORD)
    {
        VecDeriv* va = m->getVecDeriv(ops[0].second[1].first.index);
        VecDeriv* vv = m->getVecDeriv(ops[0].first.index);
        VecCoord* vx = m->getVecCoord(ops[1].first.index);
        const unsigned int n = vx->size();
        const double f_v_v = ops[0].second[0].second;
        const double f_v_a = ops[0].second[1].second;
        const double f_x_x = ops[1].second[0].second;
        const double f_x_v = ops[1].second[1].second;
        Kernels::vIntegrate(n, va->deviceRead(), vv->deviceWrite(), vx->deviceWrite(), (Real)f_v_v, (Real)f_v_a, (Real)f_x_x, (Real)f_x_v);
    }
    // optimize common CG step: x += a*p, q -= a*v
    else if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].second[0].second == 1.0 && ops[0].first.type == sofa::core::V_DERIV && ops[0].second[1].first.type == sofa::core::V_DERIV
            && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[1].second[0].second == 1.0 && ops[1].first.type == sofa::core::V_DERIV && ops[1].second[1].first.type == sofa::core::V_DERIV)
    {
        VecDeriv* vv1 = m->getVecDeriv(ops[0].second[1].first.index);
        VecDeriv* vres1 = m->getVecDeriv(ops[0].first.index);
        VecDeriv* vv2 = m->getVecDeriv(ops[1].second[1].first.index);
        VecDeriv* vres2 = m->getVecDeriv(ops[1].first.index);
        const unsigned int n = vres1->size();
        const double f1 = ops[0].second[1].second;
        const double f2 = ops[1].second[1].second;
        Kernels::vPEqBF2(n, vres1->deviceWrite(), vv1->deviceRead(), f1, vres2->deviceWrite(), vv2->deviceRead(), f2);
    }
    // optimize a pair of generic vOps
    else if (ops.size()==2 && ops[0].second.size()==2 && ops[0].second[0].second == 1.0 && ops[1].second.size()==2 && ops[1].second[0].second == 1.0)
    {
        const unsigned int n = m->getSize();
        Kernels::vOp2(n,
                (ops[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].first.index)->deviceWrite() : m->getVecDeriv(ops[0].first.index)->deviceWrite(),
                (ops[0].second[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[0].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[0].first.index)->deviceRead(),
                (ops[0].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[1].first.index)->deviceRead(),
                ops[0].second[1].second,
                (ops[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].first.index)->deviceWrite() : m->getVecDeriv(ops[1].first.index)->deviceWrite(),
                (ops[1].second[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[0].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[0].first.index)->deviceRead(),
                (ops[1].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[1].first.index)->deviceRead(),
                ops[1].second[1].second);
    }
    // optimize a pair of 4-way accumulations (such as at the end of RK4)
    else if (ops.size()==2 && ops[0].second.size()==5 && ops[0].second[0].first == ops[0].first && ops[0].second[0].second == 1.0 &&
            ops[1].second.size()==5 && ops[1].second[0].first == ops[1].first && ops[1].second[0].second == 1.0)
    {
        const unsigned int n = m->getSize();
        Kernels::vPEq4BF2(n,
                (ops[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].first.index)->deviceWrite() : m->getVecDeriv(ops[0].first.index)->deviceWrite(),
                (ops[0].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[1].first.index)->deviceRead(),
                ops[0].second[1].second,
                (ops[0].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[2].first.index)->deviceRead(),
                ops[0].second[2].second,
                (ops[0].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[3].first.index)->deviceRead(),
                ops[0].second[3].second,
                (ops[0].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[4].first.index)->deviceRead(),
                ops[0].second[4].second,
                (ops[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].first.index)->deviceWrite() : m->getVecDeriv(ops[1].first.index)->deviceWrite(),
                (ops[1].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[1].first.index)->deviceRead(),
                ops[1].second[1].second,
                (ops[1].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[2].first.index)->deviceRead(),
                ops[1].second[2].second,
                (ops[1].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[3].first.index)->deviceRead(),
                ops[1].second[3].second,
                (ops[1].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[4].first.index)->deviceRead(),
                ops[1].second[4].second);
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
            std::cout << endl;
        }
        {
            using namespace sofa::core::behavior;
            m->BaseMechanicalState::vMultiOp(ops);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
double MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::vDot(Main* m, VecId a, VecId b, bool prefetch)
{
    DEBUG_TEXT(" MechanicalObjectInternalData::vDot ");
    if (prefetch)
    {
        if (!Kernels::supportMultiVDot()) {DEBUG_TEXT("~MechanicalObjectInternalData::vDot "); return 0.0;} // no kernel available for combining multiple operations
        m->data.preVDot.a = a;
        m->data.preVDot.b = b;
        if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
            m->data.preVDot.size = m->getVecCoord(a.index)->size();
        else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
            m->data.preVDot.size = m->getVecDeriv(a.index)->size();
        else {DEBUG_TEXT("~MechanicalObjectInternalData::vDot"); return 0.0;} // invalid operation -> ignore prefetching
        if (m->data.preVDot.size == 0) return 0.0; // empty operation -> ignore prefetching

        m->data.preVDot.id = m->data.preVDot.objects().size();
        m->data.preVDot.objects().push_back(m);
        DEBUG_TEXT("~MechanicalObjectInternalData::vDot ");
        return 0.0;
    }
    else if (m->data.preVDot.id >= 0)
    {
        helper::vector<Main*>& objects = m->data.preVDot.objects();
        if (!objects.empty())
        {
            if (objects.size() == 1)
            {
                // only one object -> use regular kernel
                m->data.preVDot.id = -1;
            }
            else //if (objects.size() > 1)
            {
                //std::cout << "PREFETCH VDOT: " << m->data.preVDot.objects().size() << " objects" << std::endl;
                helper::vector<VDotOp> ops(objects.size());
                helper::vector<double> results(objects.size());
                for (unsigned int i=0; i<objects.size(); ++i)
                {
                    VecId a = objects[i]->data.preVDot.a;
                    VecId b = objects[i]->data.preVDot.b;
                    ops[i].a = (a.type == sofa::core::V_COORD) ? objects[i]->getVecCoord(a.index)->deviceRead() : objects[i]->getVecDeriv(a.index)->deviceRead();
                    ops[i].b = (b.type == sofa::core::V_COORD) ? objects[i]->getVecCoord(b.index)->deviceRead() : objects[i]->getVecDeriv(b.index)->deviceRead();
                    ops[i].size = objects[i]->data.preVDot.size;
                    results[i] = 0.0;
                }
                unsigned int nmax = (unsigned int)myopenclMultiOpMax;
                for (unsigned int i0 = 0; i0 < ops.size();)
                {
                    unsigned int n = (ops.size()-i0 > nmax) ? nmax : ops.size()-i0;
                    int tmpsize = Kernels::multiVDotTmpSize(n, &(ops[i0]));
                    if (tmpsize == 0)
                    {
                        Kernels::multiVDot(n, &(ops[i0]), &(results[i0]), _device_pointer(), NULL);
                    }
                    else
                    {
                        m->data.tmpdot.recreate(tmpsize);
                        Kernels::multiVDot(n, &(ops[i0]), &(results[i0]), m->data.tmpdot.deviceWrite(), (Real*)(&(m->data.tmpdot.getCached(0))));
                    }
                    i0 += n;
                }
                for (unsigned int i=0; i<objects.size(); ++i)
                {
                    objects[i]->data.preVDot.result = results[i];
                }
            }
            objects.clear();
        }
        if (m->data.preVDot.id != -1) // prefetching was done
        {
            m->data.preVDot.id = -1;
            DEBUG_TEXT("~MechanicalObjectInternalData::vDot ");
            return m->data.preVDot.result;
        }
    }
    Real r = 0.0f;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        VecCoord* va = m->getVecCoord(a.index);
        VecCoord* vb = m->getVecCoord(b.index);
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
        VecDeriv* va = m->getVecDeriv(a.index);
        VecDeriv* vb = m->getVecDeriv(b.index);
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
void MechanicalObjectInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::resetForce(Main* m, bool prefetch)
{
    DEBUG_TEXT("*MechanicalObjectInternalData::resetForce ");
    VecDeriv& f= *m->getF();
    if (f.size() == 0) return;
    if (prefetch)
    {
        if (!Kernels::supportMultiVClear()) return; // no kernel available for combining multiple operations
        m->data.preVResetForce.size = f.size();
        m->data.preVResetForce.id = m->data.preVResetForce.objects().size();
        m->data.preVResetForce.objects().push_back(m);
        return;
    }
    else if (m->data.preVResetForce.id >= 0)
    {
        helper::vector<Main*>& objects = m->data.preVResetForce.objects();
        if (!objects.empty())
        {
            if (objects.size() == 1)
            {
                // only one operation -> use regular kernel
                m->data.preVResetForce.id = -1;
            }
            else
            {
                int nops = objects.size();
                helper::vector< VClearOp > ops(nops);
                for (unsigned int i=0; i<objects.size(); ++i)
                {
                    Main* o = objects[i];
                    ops[i].res = o->getF()->deviceWrite();
                    ops[i].size = o->data.preVResetForce.size;
                }
                Kernels::multiVClear(nops, &(ops[0]));
            }
            objects.clear();
        }
        if (m->data.preVResetForce.id != -1) // prefetching was done
        {
            m->data.preVResetForce.id = -1;
            return;
        }
    }
    Kernels::vClear(f.size(), f.deviceWrite());

//	exit(1);
}


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLMechanicalObject_ImplMethods(T)										\
	template<> double MechanicalObject< T >::vDot(VecId a, VecId b)					\
	{ return data.vDot(this, a, b, this->isPrefetching()); }						\
	template<> void MechanicalObject< T >::vOp(VecId v, VecId a, VecId b, double f) \
	{ data.vOp(this, v, a, b, f, this->isPrefetching()); }							\
	template<> bool MechanicalObject< T >::canPrefetch() const						\
	{ return true; }																\
	template<> void MechanicalObject< T >::accumulateForce()						\
	{ data.accumulateForce(this, this->isPrefetching()); }							\
	template<> void MechanicalObject< T >::addDxToCollisionModel()					\
	{ data.addDxToCollisionModel(this, this->isPrefetching()); }					\
	template<> void MechanicalObject< T >::resetForce()								\
	{ data.resetForce(this, this->isPrefetching()); }								\
	template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops)			\
	{ data.vMultiOp(this, ops, this->isPrefetching()); }							\
//*/

OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3fTypes);
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3f1Types);
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3dTypes);
OpenCLMechanicalObject_ImplMethods(gpu::opencl::OpenCLVec3d1Types);

#undef OpenCLMechanicalObject_ImplMethods


} // namespace container

} // namespace component

} // namespace sofa

#undef DEBUG_TEXT

#endif
