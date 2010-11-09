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
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL

#include "CudaMechanicalObject.h"
#include <sofa/component/container/MechanicalObject.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);


    struct VDotOp
    {
        const void* a;
        const void* b;
        int size;
    };

    int MultiMechanicalObjectCudaVec3f_vDotTmpSize(unsigned int n, VDotOp* ops);
    void MultiMechanicalObjectCudaVec3f_vDot(unsigned int n, VDotOp* ops, double* results, void* tmp, float* cputmp);

    struct VOpF
    {
        void* res;
        const void* a;
        const void* b;
        float f;
        int size;
    };

    struct VOpD
    {
        void* res;
        const void* a;
        const void* b;
        double f;
        int size;
    };

    void MultiMechanicalObjectCudaVec3f_vOp(unsigned int n, VOpF* ops);

    struct VClearOp
    {
        void* res;
        int size;
    };

    void MultiMechanicalObjectCudaVec3f_vClear(unsigned int n, VClearOp* ops);

    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f1_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f1_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f1_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec3f1_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec3f1_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec3f1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec3f1_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f1_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);



#ifdef SOFA_GPU_CUDA_DOUBLE

    void MechanicalObjectCudaVec3d_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3d_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3d_vMEq(unsigned int size, void* res, double f);
    void MechanicalObjectCudaVec3d_vEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec3d_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3d_vPEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec3d_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3d_vOp(unsigned int size, void* res, const void* a, const void* b, double f);
    void MechanicalObjectCudaVec3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
    void MechanicalObjectCudaVec3d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2);
    void MechanicalObjectCudaVec3d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24);
    void MechanicalObjectCudaVec3d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2);
    int MechanicalObjectCudaVec3d_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);

    void MechanicalObjectCudaVec3d1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3d1_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3d1_vMEq(unsigned int size, void* res, double f);
    void MechanicalObjectCudaVec3d1_vEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec3d1_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3d1_vPEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec3d1_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3d1_vOp(unsigned int size, void* res, const void* a, const void* b, double f);
    void MechanicalObjectCudaVec3d1_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
    void MechanicalObjectCudaVec3d1_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2);
    void MechanicalObjectCudaVec3d1_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24);
    void MechanicalObjectCudaVec3d1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2);
    int MechanicalObjectCudaVec3d1_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3d1_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);



#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"

template<>
class CudaKernelsMechanicalObject<CudaVec3fTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3f_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec3f_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaVec3f_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec3f_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3f_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec3f_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec3f_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaVec3f_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectCudaVec3f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
    {   MechanicalObjectCudaVec3f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
    {
        MechanicalObjectCudaVec3f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
    {   MechanicalObjectCudaVec3f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec3f_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaVec3f_vDot(size, res, a, b, tmp, cputmp); }
    static bool supportMultiVDot() { return mycudaMultiOpMax>0; }
    static int multiVDotTmpSize(unsigned int n, VDotOp* ops)
    {   return MultiMechanicalObjectCudaVec3f_vDotTmpSize(n, ops); }
    static void multiVDot(unsigned int n, VDotOp* ops, double* results, void* tmp, float* cputmp)
    {   MultiMechanicalObjectCudaVec3f_vDot(n, ops, results, tmp, cputmp); }
    typedef VOpF VOp;
    static bool supportMultiVOp() { return mycudaMultiOpMax>0; }
    static void multiVOp(unsigned int n, VOp* ops)
    {   MultiMechanicalObjectCudaVec3f_vOp(n, ops); }
    static bool supportMultiVClear() { return mycudaMultiOpMax>0; }
    static void multiVClear(unsigned int n, VClearOp* ops)
    {   MultiMechanicalObjectCudaVec3f_vClear(n, ops); }
};

template<>
class CudaKernelsMechanicalObject<CudaVec3f1Types>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3f1_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec3f1_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaVec3f1_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec3f1_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3f1_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec3f1_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec3f1_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaVec3f1_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectCudaVec3f1_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
    {   MechanicalObjectCudaVec3f1_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
    {
        MechanicalObjectCudaVec3f1_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
    {   MechanicalObjectCudaVec3f1_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec3f1_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaVec3f1_vDot(size, res, a, b, tmp, cputmp); }
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(unsigned int, VDotOp*)
    {   return 0; }
    static void multiVDot(unsigned int, VDotOp*, double*, void*, float*)
    {}
    typedef VOpF VOp;
    static bool supportMultiVOp() { return false /*mycudaMultiOpMax>0*/; }
    static void multiVOp(unsigned int /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectCudaVec3f1_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(unsigned int, VClearOp*)
    {}
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsMechanicalObject<CudaVec3dTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3d_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec3d_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaVec3d_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec3d_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3d_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec3d_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec3d_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaVec3d_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
    {   MechanicalObjectCudaVec3d_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
    {   MechanicalObjectCudaVec3d_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
    {
        MechanicalObjectCudaVec3d_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
    {   MechanicalObjectCudaVec3d_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec3d_vDotTmpSize(size); }
    static void vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp)
    {   MechanicalObjectCudaVec3d_vDot(size, res, a, b, tmp, cputmp); }
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(unsigned int, VDotOp*)
    {   return 0; }
    static void multiVDot(unsigned int, VDotOp*, double*, void*, double*)
    {}
    typedef VOpD VOp;
    static bool supportMultiVOp() { return false /*mycudaMultiOpMax>0*/; }
    static void multiVOp(unsigned int /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectCudaVec3d_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(unsigned int, VClearOp*)
    {}
};

template<>
class CudaKernelsMechanicalObject<CudaVec3d1Types>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3d1_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec3d1_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaVec3d1_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec3d1_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec3d1_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec3d1_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec3d1_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaVec3d1_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
    {   MechanicalObjectCudaVec3d1_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
    {   MechanicalObjectCudaVec3d1_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
    {
        MechanicalObjectCudaVec3d1_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
    {   MechanicalObjectCudaVec3d1_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec3d1_vDotTmpSize(size); }
    static void vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp)
    {   MechanicalObjectCudaVec3d1_vDot(size, res, a, b, tmp, cputmp); }
    static bool supportMultiVDot() { return false; }
    static int multiVDotTmpSize(unsigned int, VDotOp*)
    {   return 0; }
    static void multiVDot(unsigned int, VDotOp*, double*, void*, double*)
    {}
    typedef VOpD VOp;
    static bool supportMultiVOp() { return false /*mycudaMultiOpMax>0*/; }
    static void multiVOp(unsigned int /*n*/, VOp* /*ops*/)
    {   /*MultiMechanicalObjectCudaVec3d1_vOp(n, ops);*/ }
    static bool supportMultiVClear() { return false; }
    static void multiVClear(unsigned int, VClearOp*)
    {}
};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu

namespace component
{

namespace container
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::accumulateForce(Main* m, bool prefetch)
{
    if (prefetch) return;
    if (!m->externalForces.getValue().empty())
    {
        //std::cout << "ADD: external forces, size = "<< m->externalForces->size() << std::endl;
        Kernels::vAssign(m->externalForces.getValue().size(),m->f.beginEdit()->deviceWrite(),m->externalForces.getValue().deviceRead());
        m->f.endEdit();
    }
    //else std::cout << "NO external forces" << std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDxToCollisionModel(Main* m, bool prefetch)
{
    if (prefetch) return;
    Kernels::vAdd(m->xfree.getValue().size(), m->x.beginEdit()->deviceWrite(), m->xfree.getValue().deviceRead(), m->dx.getValue().deviceRead());
    m->x.endEdit();
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vAlloc(Main* m, VecId v)
{
    if (v.type == sofa::core::V_COORD && v.index >= VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = m->getVecCoord(v.index);
        vec->recreate(m->vsize);
    }
    else if (v.type == sofa::core::V_DERIV && v.index >= VecDerivId::V_FIRST_DYNAMIC_INDEX)
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
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, double f, bool prefetch)
{
    if (prefetch)
    {
        if (!Kernels::supportMultiVOp()) return; // no kernel available for combining multiple operations
        if (v.isNull() || a.isNull() || b.isNull()) return; // ignore invalid or simple operations
        VOp op;
        op.v = v;
        op.a = a;
        op.b = b;
        op.f = f;
        //op.size = (a.type == sofa::core::V_COORD) ? m->getVecCoord(a.index)->size() : m->getVecDeriv(a.index)->size();
        op.size = m->getSize();
        m->data.preVOp.push_back(op);
        m->data.preVOp.id = m->data.preVOp.objects().size();
        m->data.preVOp.objects().push_back(m);
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
                        //ops[nops].res = (oops[j].v.type == sofa::core::V_COORD) ? o->getVecCoord(v.index)->deviceWrite() : o->getVecDeriv(v.index)->deviceWrite();
                        //ops[nops].a = (oops[j].a.type == sofa::core::V_COORD) ? o->getVecCoord(a.index)->deviceRead() : o->getVecDeriv(a.index)->deviceRead();
                        //ops[nops].b = (oops[j].b.type == sofa::core::V_COORD) ? o->getVecCoord(b.index)->deviceRead() : o->getVecDeriv(b.index)->deviceRead();
                        ops[nops].res = (oops[j].v.type == sofa::core::V_COORD) ? o->write((VecCoordId)v)->beginEdit()->deviceWrite() : o->write((VecDerivId)v)->beginEdit()->deviceWrite();
                        ops[nops].a = (oops[j].a.type == sofa::core::V_COORD) ? o->read((ConstVecCoordId)a)->getValue().deviceRead() : o->read((ConstVecDerivId)a)->getValue().deviceRead();
                        ops[nops].b = (oops[j].b.type == sofa::core::V_COORD) ? o->read((ConstVecCoordId)b)->getValue().deviceRead() : o->read((ConstVecDerivId)b)->getValue().deviceRead();
                        ops[nops].f = (Real)oops[j].f;
                        ops[nops].size = oops[j].size;
                        ++nops;
                    }
                }
                Kernels::multiVOp(nops, &(ops[0]));

                //Do not forget to end edition of Data
                for (unsigned int i=0; i<objects.size(); ++i)
                {
                    Main* o = objects[i];
                    helper::vector<VOp>& oops = o->data.preVOp;
                    for (unsigned int j=0; j<oops.size(); ++j)
                    {
                        (oops[j].v.type == sofa::core::V_COORD) ? o->write((VecCoordId)v)->endEdit() : o->write((VecDerivId)v)->endEdit();
                    }
                }
            }
            objects.clear();
        }
        if (m->data.preVOp.id != -1) // prefetching was done
        {
            m->data.preVOp.resize(m->data.preVOp.size()-1);
            if (m->data.preVOp.empty())
                m->data.preVOp.id = -1;
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
                Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                VecCoord* vv = d_vv->beginEdit();
                vv->recreate(m->vsize);
                Kernels::vClear(vv->size(), vv->deviceWrite());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
                VecDeriv* vv = d_vv->beginEdit();
                vv->recreate(m->vsize);
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
                    Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                    VecCoord* vv = d_vv->beginEdit();
                    Kernels::vMEq(vv->size(), vv->deviceWrite(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
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
                    Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                    const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                    VecCoord* vv = d_vv->beginEdit();
                    const VecCoord* vb = &d_vb->getValue();
                    vv->recreate(vb->size());
                    Kernels::vEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
                    const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
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
                Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                const Data<VecCoord>* d_va = m->read((ConstVecCoordId)a);
                VecCoord* vv = d_vv->beginEdit();
                const VecCoord* va = &d_va->getValue();
                vv->recreate(va->size());
                Kernels::vAssign(vv->size(), vv->deviceWrite(), va->deviceRead());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
                const Data<VecDeriv>* d_va = m->read((ConstVecDerivId)a);
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
                        Data<VecDeriv>* d_vv = m->write((VecCoordId)v);
                        VecDeriv* vv = d_vv->beginEdit();
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
                        const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
                        VecDeriv* vv = d_vv->beginEdit();
                        const VecDeriv* vb = &d_vb->getValue();
                        vv->resize(vb->size());
                        Kernels::vPEq(vv->size(), vv->deviceWrite(), vb->deviceRead());
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
                        Data<VecDeriv>* d_vv = m->write((VecCoordId)v);
                        VecDeriv* vv = d_vv->beginEdit();
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEqBF(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecDeriv>* d_vv = m->write((VecDerivId)v);
                        const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
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
                        Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                        const Data<VecCoord>* d_va = m->read((ConstVecCoordId)a);
                        VecCoord* vv = d_vv->beginEdit();
                        const VecCoord* va = &d_va->getValue();
                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vAdd(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                        VecCoord* vv = d_vv->beginEdit();
                        const Data<VecCoord>* d_va = m->read((ConstVecCoordId)a);
                        const VecCoord* va = &d_va->getValue();
                        const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                        const VecCoord* vb = &d_vb->getValue();
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
                        Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                        const Data<VecCoord>* d_va = m->read((ConstVecCoordId)a);
                        VecCoord* vv = d_vv->beginEdit();
                        const VecCoord* va = &d_va->getValue();

                        vv->recreate(va->size());
                        if (b.type == sofa::core::V_COORD)
                        {
                            const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                            const VecCoord* vb = &d_vb->getValue();
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vOp(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }

                        d_vv->endEdit();
                    }
                    else if (b.type == sofa::core::V_DERIV)
                    {
                        Data<VecCoord>* d_vv = m->write((VecCoordId)v);
                        VecCoord* vv = d_vv->beginEdit();
                        const Data<VecCoord>* d_va = m->read((ConstVecCoordId)a);
                        const VecCoord* va = &d_va->getValue();
                        const Data<VecCoord>* d_vb = m->read((ConstVecCoordId)b);
                        const VecCoord* vb = &d_vb->getValue();
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
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vMultiOp(Main* m, const VMultiOp& ops, bool prefetch)
{
    if (prefetch) return;
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
        const Data<VecDeriv>* d_va = m->read(ConstVecDerivId(ops[0].second[1].first.getId(m)));
        const VecDeriv* va = &d_va->getValue();
        Data<VecDeriv>* d_vv = m->write(VecDerivId(ops[0].first.getId(m)));
        VecDeriv* vv = d_vv->beginEdit();
        Data<VecCoord>* d_vx = m->write(VecCoordId(ops[1].first.getId(m)));
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
        const Data<VecDeriv>* d_vv1 = m->read(ConstVecDerivId(ops[0].second[1].first.getId(m)));
        const VecDeriv* vv1 = &d_vv1->getValue();
        const Data<VecDeriv>* d_vv2 = m->read(ConstVecDerivId(ops[1].second[1].first.getId(m)));
        const VecDeriv* vv2 = &d_vv2->getValue();

        Data<VecDeriv>* d_vres1 = m->write(VecDerivId(ops[0].first.getId(m)));
        VecDeriv* vres1 = d_vres1->beginEdit();
        Data<VecDeriv>* d_vres2 = m->write(VecDerivId(ops[1].first.getId(m)));
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

        void* w0Ptr, *r0Ptr0, *r0Ptr1;
        void* w1Ptr, *r1Ptr0, *r1Ptr1;

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        r0Ptr0 = (ops[0].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[0].second[0].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[0].second[0].first.getId(m)))->getValue().deviceRead();
        r0Ptr1 = (ops[0].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[0].second[1].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[0].second[1].first.getId(m)))->getValue().deviceRead();
        w1Ptr  = (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[1].first.getId(m)))->beginEdit()->deviceWrite() : m->write(VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();
        r1Ptr0 = (ops[1].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[1].second[0].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[1].second[0].first.getId(m)))->getValue().deviceRead();
        r1Ptr1 = (ops[1].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[1].second[1].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[1].second[1].first.getId(m)))->getValue().deviceRead();

        Kernels::vOp2(n, w0Ptr, r0Ptr0, r0Ptr1,	ops[0].second[1].second, w1Ptr, r1Ptr0, r1Ptr1, ops[1].second[1].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(VecDerivId(ops[0].first.getId(m)))->endEdit();
        (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[1].first.getId(m)))->endEdit() : m->write(VecDerivId(ops[1].first.getId(m)))->endEdit();
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

        void* w0Ptr, *r0Ptr[4];
        void* w1Ptr, *r1Ptr[4];

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        w1Ptr  = (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[1].first.getId(m)))->beginEdit()->deviceWrite() : m->write(VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();

        for(unsigned int i=0 ; i < 4 ; i++)
        {
            r0Ptr[i] = (ops[0].second[i+1].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[0].second[i+1].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[0].second[i+1].first.getId(m)))->getValue().deviceRead();
            r1Ptr[i] = (ops[1].second[i+1].first.getId(m).type == sofa::core::V_COORD) ? m->read(ConstVecCoordId(ops[1].second[i+1].first.getId(m)))->getValue().deviceRead() : m->read(ConstVecDerivId(ops[1].second[i+1].first.getId(m)))->getValue().deviceRead();;

        }
        Kernels::vPEq4BF2(n, w0Ptr, r0Ptr[0], ops[0].second[1].second, r0Ptr[1], ops[0].second[2].second, r0Ptr[2], ops[0].second[3].second, r0Ptr[3], ops[0].second[4].second,
                w1Ptr, r1Ptr[0], ops[1].second[1].second, r1Ptr[1], ops[1].second[2].second, r1Ptr[2], ops[1].second[3].second, r1Ptr[3], ops[1].second[4].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(VecDerivId(ops[0].first.getId(m)))->endEdit();
        (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(VecCoordId(ops[1].first.getId(m)))->endEdit() : m->write(VecDerivId(ops[1].first.getId(m)))->endEdit();

        //Kernels::vPEq4BF2(n,
        //	(ops[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].first.index)->deviceWrite() : m->getVecDeriv(ops[0].first.index)->deviceWrite(),
        //	(ops[0].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[1].first.index)->deviceRead(),
        //	ops[0].second[1].second,
        //	(ops[0].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[2].first.index)->deviceRead(),
        //	ops[0].second[2].second,
        //	(ops[0].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[3].first.index)->deviceRead(),
        //	ops[0].second[3].second,
        //	(ops[0].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[4].first.index)->deviceRead(),
        //	ops[0].second[4].second,
        //	(ops[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].first.index)->deviceWrite() : m->getVecDeriv(ops[1].first.index)->deviceWrite(),
        //	(ops[1].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[1].first.index)->deviceRead(),
        //	ops[1].second[1].second,
        //	(ops[1].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[2].first.index)->deviceRead(),
        //	ops[1].second[2].second,
        //	(ops[1].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[3].first.index)->deviceRead(),
        //	ops[1].second[3].second,
        //	(ops[1].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[4].first.index)->deviceRead(),
        //	ops[1].second[4].second);
    }
    else // no optimization for now for other cases
    {
        std::cout << "CUDA: unoptimized vMultiOp:"<<std::endl;
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
double MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vDot(Main* m, ConstVecId a, ConstVecId b, bool prefetch)
{
    if (prefetch)
    {
        if (!Kernels::supportMultiVDot()) return 0.0; // no kernel available for combining multiple operations
        m->data.preVDot.a = a;
        m->data.preVDot.b = b;

        //if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
        //	m->data.preVDot.size = m->getVecCoord(a.index)->size();
        //else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
        //	m->data.preVDot.size = m->getVecDeriv(a.index)->size();
        if ( (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD) || (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV))
            m->data.preVDot.size = m->getSize();
        else return 0.0; // invalid operation -> ignore prefetching
        if (m->data.preVDot.size == 0) return 0.0; // empty operation -> ignore prefetching

        m->data.preVDot.id = m->data.preVDot.objects().size();
        m->data.preVDot.objects().push_back(m);
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
                    ConstVecId a = objects[i]->data.preVDot.a;
                    ConstVecId b = objects[i]->data.preVDot.b;
                    ops[i].a = (a.type == sofa::core::V_COORD) ? objects[i]->read(ConstVecCoordId(a))->getValue().deviceRead() : objects[i]->read(ConstVecDerivId(a))->getValue().deviceRead();
                    ops[i].b = (b.type == sofa::core::V_COORD) ? objects[i]->read(ConstVecCoordId(b))->getValue().deviceRead() : objects[i]->read(ConstVecDerivId(b))->getValue().deviceRead();
                    ops[i].size = objects[i]->data.preVDot.size;
                    results[i] = 0.0;
                }
                unsigned int nmax = (unsigned int)mycudaMultiOpMax;
                for (unsigned int i0 = 0; i0 < ops.size();)
                {
                    unsigned int n = (ops.size()-i0 > nmax) ? nmax : ops.size()-i0;
                    int tmpsize = Kernels::multiVDotTmpSize(n, &(ops[i0]));
                    if (tmpsize == 0)
                    {
                        Kernels::multiVDot(n, &(ops[i0]), &(results[i0]), NULL, NULL);
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
            return m->data.preVDot.result;
        }
    }
    Real r = 0.0f;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        const VecCoord* va = &m->read(ConstVecCoordId(a))->getValue();
        const VecCoord* vb = &m->read(ConstVecCoordId(b))->getValue();
        int tmpsize = Kernels::vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
        }
        else
        {
            m->data.tmpdot.recreate(tmpsize);
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), m->data.tmpdot.deviceWrite(), (Real*)(&(m->data.tmpdot.getCached(0))));
        }
    }
    else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
    {
        const VecDeriv* va = &m->read(ConstVecDerivId(a))->getValue();
        const VecDeriv* vb = &m->read(ConstVecDerivId(b))->getValue();
        int tmpsize = Kernels::vDotTmpSize(va->size());
        if (tmpsize == 0)
        {
            Kernels::vDot(va->size(), &r, va->deviceRead(), vb->deviceRead(), NULL, NULL);
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
        //std::cout << "CUDA vDot: GPU="<<r<<"  CPU="<<r2<<" relative error="<<(fabsf(r2)>0.000001?fabsf(r-r2)/fabsf(r2):fabsf(r-r2))<<"\n";
#endif
    }
    else
    {
        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
    }
    return r;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::resetForce(Main* m, bool prefetch)
{
    Data<VecDeriv>* d_f = m->write(VecDerivId::force());
    VecDeriv& f = *d_f->beginEdit();

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
                    ops[i].res = f.deviceWrite();
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
    d_f->endEdit();
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyToBaseVector(Main* m, defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        const VecCoord& vSrc = m->read(ConstVecCoordId(src))->getValue();

        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = dest->size()/coordDim;

        for (unsigned int i=0; i<nbEntries; ++i)
            for (unsigned int j=0; j<coordDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vSrc[offset + i],j,tmp);
                dest->set(i * coordDim + j, tmp);
            }
        // offset += vSrc.size() * coordDim;
    }
    else
    {
        const VecDeriv& vSrc = m->read(ConstVecDerivId(src))->getValue();

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = dest->size()/derivDim;

        for (unsigned int i=0; i<nbEntries; i++)
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vSrc[i + offset],j,tmp);
                dest->set(i * derivDim + j, tmp);
            }
        // offset += vSrc.size() * derivDim;
    }
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyToCudaBaseVector(Main* m, sofa::gpu::cuda::CudaBaseVector<Real> * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        const VecCoord& va = m->read(ConstVecCoordId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;

        Kernels::vAssign(nbEntries, dest->getCudaVector().deviceWrite(), ((Real *) va.deviceRead())+(offset*elemDim));
    }
    else
    {
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        const VecCoord& va = m->read(ConstVecDerivId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;

        Kernels::vAssign(nbEntries, dest->getCudaVector().deviceWrite(), ((Real *) va.deviceRead())+(offset*elemDim));
    }
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addFromBaseVectorSameSize(Main* m, VecId dest, const defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();

        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Coord>::setValue((*vDest)[i],j,tmp + src->element(offset + i * coordDim + j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(offset + i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue((*vDest)[i], j+3, tmp);
            }
        }

        offset += vDest->size() * coordDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(VecDerivId(dest));
        VecDeriv* vDest = d_vDest->beginEdit();

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Deriv>::setValue((*vDest)[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }
        offset += vDest->size() * derivDim;
        d_vDest->endEdit();
    }
};

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addFromCudaBaseVectorSameSize(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVector<Real> *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        printf("ERROR CudaMechanicalObject::addCudaBaseVectorToState<V_COORD> NOT YET IMPLEMENTED\n");
        //unsigned int elemDim = DataTypeInfo<Coord>::size();
        //VecCoord* va = m->getVecCoord(dest.index);
        //const unsigned int nbEntries = src->size()/elemDim;

        //Kernels::vPEq(nbEntries, va->deviceWrite(), ((Real *) src->getCudaVector().deviceRead())+(offset*elemDim));

        Data<VecCoord>* d_vDest = m->write(VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Coord>::setValue((*vDest)[i],j,tmp + src->element(offset + i * coordDim + j));
            }

            helper::Quater<double> q_src;
            helper::Quater<double> q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j+3,tmp);
                q_dest[j]=tmp;
                q_src[j]=src->element(offset + i * coordDim + j+3);
            }
            //q_dest = q_dest*q_src;
            q_dest = q_src*q_dest;
            for (unsigned int j=0; j<4; j++)
            {
                Real tmp=q_dest[j];
                DataTypeInfo<Coord>::setValue((*vDest)[i], j+3, tmp);
            }
        }

        offset += vDest->size() * coordDim;
        d_vDest->endEdit();
    }
    else
    {
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        Data<VecDeriv>* d_va = m->write(VecDerivId(dest));
        VecDeriv* va = d_va->beginEdit();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vPEq(nbEntries, va->deviceWrite(), ((Real *) src->getCudaVector().deviceRead())+(offset*elemDim));

        offset += va->size() * elemDim;
        d_va->endEdit();
    }
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_ImplMethods(T) \
template<> bool MechanicalObject< T >::canPrefetch() const \
{ return true; } \
template<> void MechanicalObject< T >::accumulateForce(const core::ExecParams* /* params */) \
{ data.accumulateForce(this, this->isPrefetching()); } \
template<> void MechanicalObject< T >::vOp(core::VecId v, core::ConstVecId a, core::ConstVecId b, double f, const core::ExecParams* /* params */) \
{ data.vOp(this, v, a, b, f, this->isPrefetching()); }		\
template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops, const core::ExecParams* /* params */) \
{ data.vMultiOp(this, ops, this->isPrefetching()); } \
template<> double MechanicalObject< T >::vDot(core::ConstVecId a, core::ConstVecId b, const core::ExecParams* /* params */) \
{ return data.vDot(this, a, b, this->isPrefetching()); }				    \
template<> void MechanicalObject< T >::resetForce(const core::ExecParams* /* params */) \
{ data.resetForce(this, this->isPrefetching()); } \
template<> void MechanicalObject< T >::addDxToCollisionModel() \
{ data.addDxToCollisionModel(this, this->isPrefetching()); } \
template<> void MechanicalObject< T >::copyToBaseVector(defaulttype::BaseVector * dest, core::ConstVecId src, unsigned int &offset) \
{ if (CudaBaseVector<Real> * vec = dynamic_cast<CudaBaseVector<Real> *>(dest)) data.copyToCudaBaseVector(this, vec,src,offset); \
else data.copyToBaseVector(this, dest,src,offset); } \
template<> void MechanicalObject< T >::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector *src, unsigned int &offset) \
{ if (const CudaBaseVector<Real> * vec = dynamic_cast<const CudaBaseVector<Real> *>(src)) data.addFromCudaBaseVectorSameSize(this, dest,vec,offset); \
else data.addFromBaseVectorSameSize(this, dest,src,offset); }

CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaMechanicalObject_ImplMethods

}

} // namespace component

} // namespace sofa

#endif
