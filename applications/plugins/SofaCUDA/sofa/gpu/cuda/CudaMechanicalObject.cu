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
#include "CudaMathRigid.h"
#include "mycuda.h"
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void MechanicalObjectCudaVec1f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec1f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec1f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec1f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec1f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec1f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec1f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec1f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec1f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec1f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec1f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec1f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec1f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec1f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
    void MechanicalObjectCudaVec1f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);
    void MechanicalObjectCudaVec1f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14);
    void MechanicalObjectCudaVec1f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31);


    void MechanicalObjectCudaVec2f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec2f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec2f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec2f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec2f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec2f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec2f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec2f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec2f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec2f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec2f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec2f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec2f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec2f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
    void MechanicalObjectCudaVec2f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);
    void MechanicalObjectCudaVec2f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14);
    void MechanicalObjectCudaVec2f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31);

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
    void MechanicalObjectCudaVec3f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);
    void MechanicalObjectCudaVec3f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14);
    void MechanicalObjectCudaVec3f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31);

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
//void MechanicalObjectCudaVec3f1_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);

    void MechanicalObjectCudaVec6f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec6f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec6f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec6f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec6f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec6f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec6f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec6f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec6f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec6f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec6f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec6f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
//void MechanicalObjectCudaVec6f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);

    void MechanicalObjectCudaRigid3f_vAssignCoord(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3f_vAssignDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3f_vClearCoord(unsigned int size, void* res);
    void MechanicalObjectCudaRigid3f_vClearDeriv(unsigned int size, void* res);
    void MechanicalObjectCudaRigid3f_vMEqCoord(unsigned int size, void* res, float f);
    void MechanicalObjectCudaRigid3f_vMEqDeriv(unsigned int size, void* res, float f);
    void MechanicalObjectCudaRigid3f_vEqBFCoord(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vEqBFDeriv(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vPEqCoord(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3f_vPEqCoordDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3f_vPEqDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3f_vPEqBFCoord(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vPEqBFDeriv(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vAddCoord(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3f_vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3f_vAddDeriv(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3f_vOpCoord(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaRigid3f_vOpDeriv(unsigned int size, void* res, const void* a, const void* b, float f);
// void MechanicalObjectCudaRigid3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
//void MechanicalObjectCudaRigid3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
//void MechanicalObjectCudaRigid3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
//                                                            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
//void MechanicalObjectCudaRigid3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaRigid3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaRigid3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
//void MechanicalObjectCudaRigid3f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void MechanicalObjectCudaVec2d_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec2d_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec2d_vMEq(unsigned int size, void* res, double f);
    void MechanicalObjectCudaVec2d_vEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec2d_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec2d_vPEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec2d_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec2d_vOp(unsigned int size, void* res, const void* a, const void* b, double f);
    void MechanicalObjectCudaVec2d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
    void MechanicalObjectCudaVec2d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2);
    void MechanicalObjectCudaVec2d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24);
    void MechanicalObjectCudaVec2d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2);
    int MechanicalObjectCudaVec2d_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec2d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);
    void MechanicalObjectCudaVec2d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);
    void MechanicalObjectCudaVec2d_vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14);
    void MechanicalObjectCudaVec2d_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31);

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
    void MechanicalObjectCudaVec3d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);
    void MechanicalObjectCudaVec3d_vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14);
    void MechanicalObjectCudaVec3d_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31);

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
//void MechanicalObjectCudaVec3d1_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);

    void MechanicalObjectCudaVec6d_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6d_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec6d_vMEq(unsigned int size, void* res, double f);
    void MechanicalObjectCudaVec6d_vEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec6d_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6d_vPEqBF(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaVec6d_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec6d_vOp(unsigned int size, void* res, const void* a, const void* b, double f);
//    void MechanicalObjectCudaVec6d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v);
//    void MechanicalObjectCudaVec6d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2);
//    void MechanicalObjectCudaVec6d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
//    	void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24);
//    void MechanicalObjectCudaVec6d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2);
    int MechanicalObjectCudaVec6d_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec6d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);
//void MechanicalObjectCudaVec6d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);

    void MechanicalObjectCudaRigid3d_vAssignCoord(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vAssignDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vClearCoord(unsigned int size, void* res);
    void MechanicalObjectCudaRigid3d_vClearDeriv(unsigned int size, void* res);
    void MechanicalObjectCudaRigid3d_vMEqCoord(unsigned int size, void* res, double f);
    void MechanicalObjectCudaRigid3d_vMEqDeriv(unsigned int size, void* res, double f);
    void MechanicalObjectCudaRigid3d_vEqBFCoord(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vEqBFDeriv(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vPEqCoord(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vPEqCoordDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vPEqDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vPEqBFCoord(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vPEqBFDeriv(unsigned int size, void* res, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vAddCoord(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3d_vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3d_vAddDeriv(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaRigid3d_vOpCoord(unsigned int size, void* res, const void* a, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, double f);
    void MechanicalObjectCudaRigid3d_vOpDeriv(unsigned int size, void* res, const void* a, const void* b, double f);
// void MechanicalObjectCudaRigid3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
//void MechanicalObjectCudaRigid3d_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
//void MechanicalObjectCudaRigid3d_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
//                                                            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
//void MechanicalObjectCudaRigid3d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaRigid3d_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaRigid3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);
//void MechanicalObjectCudaRigid3d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void MechanicalObjectCudaVec1t_vClear_kernel(int size, real* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = 0.0f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vClear_kernel(int size, CudaVec2<real>* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = CudaVec2<real>::make(0.0f,0.0f);
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vClear_kernel(int size, CudaVec3<real>* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vClear_kernel(int size, CudaVec4<real>* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vMEq_kernel(int size, real* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vMEq_kernel(int size, real* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        //CudaVec3<real> ri = res[index];
        //ri *= f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vMEq_kernel(int size, real* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        //CudaVec3<real> ri = res[index];
        //ri *= f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vMEq_kernel(int size, CudaVec4<real>* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = res[index]*f;
        CudaVec4<real> v = res[index];
        v.x *= f;
        v.y *= f;
        v.z *= f;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vMEqCoord_kernel(int size, CudaRigidCoord3<real>* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    if (index < size)
    {
        // following RigidTypes we want to scale only the position of the center ?!

        CudaRigidCoord3<real> _res = CudaRigidCoord3<real>::make(res[index]);
        _res.pos[0] *= f;
        _res.pos[1] *= f;
        _res.pos[2] *= f;

        res[index] = _res;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vMEqDeriv_kernel(int size, real* res, real f)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vEqBF_kernel(int size, CudaVec4<real>* res, const CudaVec4<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = b[index] * f;
        CudaVec4<real> v = b[index];
        v.x *= f;
        v.y *= f;
        v.z *= f;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vEqBFCoord_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        CudaRigidCoord3<real> v = CudaRigidCoord3<real>::make(b[index]);
        v.pos[0] *= f;
        v.pos[1] *= f;
        v.pos[2] *= f;

        // res[index].pos = v.pos;
        // res[index].rot = v.rot;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vPEq_kernel(int size, real* res, const real* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] += a[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vPEq_kernel(int size, CudaVec2<real>* res, const CudaVec2<real>* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //res[index] += a[index];
        //index += BSIZE;
        //res[index] += a[index];
        CudaVec2<real> ai = a[index];
        CudaVec2<real> ri = res[index];
        ri.x += ai.x;
        ri.y += ai.y;
        res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vPEq_kernel(int size, real* res, const real* a)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        //CudaVec3<real> ai = a[index];
        //CudaVec3<real> ri = res[index];
        //ri += ai;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vPEq_kernel(int size, CudaVec4<real>* res, const CudaVec4<real>* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += a[index];
        CudaVec4<real> v = res[index];
        CudaVec4<real> v2 = a[index];
        v.x += v2.x;
        v.y += v2.y;
        v.z += v2.z;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqCoord_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
//	  CudaRigidCoord3<real> v = CudaRigidCoord3<real>::make(res[index]);
        // following RigidTypes, we do not care about orientation
        // CudaRigidCoord3<real> v = res[index];
        // CudaRigidCoord3<real> v2 = a[index];
        // v.pos[0] += v2.pos[0];
        // v.pos[1] += v2.pos[1];
        // v.pos[2] += v2.pos[2];
        // res[index] = v;
        // but we care when it is a "+" and not a "+=" so I make the version where we care :
        CudaRigidCoord3<real> v = res[index];
        CudaRigidCoord3<real> v2 = a[index];

        v += v2;

        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqCoordDeriv_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidDeriv3<real>* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
//	  CudaRigidCoord3<real> v = CudaRigidCoord3<real>::make(res[index]);
        CudaRigidCoord3<real> v = res[index];
        CudaRigidDeriv3<real> v2 = a[index];

        // use of operator + between CudaRigidCoord3 and CudaRigidDeriv3
        res[index] = v + v2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqDeriv_kernel(int size, real* res, const real* a)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vPEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vPEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vPEqBF_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vPEqBF_kernel(int size, CudaVec4<real>* res, const CudaVec4<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += b[index] * f;
        CudaVec4<real> v = res[index];
        CudaVec4<real> v2 = b[index];
        v.x += v2.x*f;
        v.y += v2.y*f;
        v.z += v2.z*f;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqBFCoord_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        CudaRigidCoord3<real> v = res[index];
        CudaRigidCoord3<real> v2 = b[index];
        v.pos[0] += v2.pos[0]*f;
        v.pos[1] += v2.pos[1]*f;
        v.pos[2] += v2.pos[2]*f;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqBFCoordDeriv_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidDeriv3<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
//	  CudaRigidCoord3<real> v = CudaRigidCoord3<real>::make(res[index]);
        CudaRigidCoord3<real> v = res[index];
        CudaRigidDeriv3<real> v2 = b[index];

        v2.pos.x *= f;
        v2.pos.y *= f;
        v2.pos.z *= f;

        v2.rot.x *= f;
        v2.rot.y *= f;
        v2.rot.z *= f;

        // v.pos[0] += v2.pos[0]*f;
        // v.pos[1] += v2.pos[1]*f;
        // v.pos[2] += v2.pos[2]*f;
        // CudaVec4<real> orient = CudaVec4<real>::make(v.rot[0]*f, v.rot[1]*f, v.rot[2]*f, v.rot[3]*f);
        // orient = orient*invnorm(orient);
        // CudaVec3<real> vOrient = CudaVec3<real>::make(a.rot.x, a.rot.y, a.rot.z);
        // CudaVec4<real> qDot = vectQuatMult(orient, vOrient);
        // orient.x += qDot.x*0.5f;
        // orient.y += qDot.y*0.5f;
        // orient.z += qDot.z*0.5f;
        // orient.w += qDot.w*0.5f;
        // orient = orient*invnorm(orient);

        // v.rot[0] = orient.x;
        // v.rot[1] = orient.y;
        // v.rot[2] = orient.z;
        // v.rot[3] = orient.w;

        res[index] = v + v2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel(int size, real* res, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vPEqBF2_kernel(int size, real* res1, const real* b1, real f1, real* res2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vPEqBF2_kernel(int size, real* res1, const real* b1, real f1, real* res2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vPEqBF2_kernel(int size, real* res1, const real* b1, real f1, real* res2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vPEqBF2_kernel(int size, CudaVec4<real>* res1, const CudaVec4<real>* b1, real f1, CudaVec4<real>* res2, const CudaVec4<real>* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += b[index] * f;
        CudaVec4<real> v = res1[index];
        CudaVec4<real> v2 = b1[index];
        v.x += v2.x*f1;
        v.y += v2.y*f1;
        v.z += v2.z*f1;
        res1[index] = v;
        v = res2[index];
        v2 = b2[index];
        v.x += v2.x*f2;
        v.y += v2.y*f2;
        v.z += v2.z*f2;
        res2[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vPEq4BF2_kernel(int size, real* res1, const real* b11, real f11, const real* b12, real f12, const real* b13, real f13, const real* b14, real f14,
        real* res2, const real* b21, real f21, const real* b22, real f22, const real* b23, real f23, const real* b24, real f24)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        real r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vPEq4BF2_kernel(int size, real* res1, const real* b11, real f11, const real* b12, real f12, const real* b13, real f13, const real* b14, real f14,
        real* res2, const real* b21, real f21, const real* b22, real f22, const real* b23, real f23, const real* b24, real f24)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        real r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vPEq4BF2_kernel(int size, real* res1, const real* b11, real f11, const real* b12, real f12, const real* b13, real f13, const real* b14, real f14,
        real* res2, const real* b21, real f21, const real* b22, real f22, const real* b23, real f23, const real* b24, real f24)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        real r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vPEq4BF2_kernel(int size, CudaVec4<real>* res1, const CudaVec4<real>* b11, real f11, const CudaVec4<real>* b12, real f12, const CudaVec4<real>* b13, real f13, const CudaVec4<real>* b14, real f14,
        CudaVec4<real>* res2, const CudaVec4<real>* b21, real f21, const CudaVec4<real>* b22, real f22, const CudaVec4<real>* b23, real f23, const CudaVec4<real>* b24, real f24)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        CudaVec4<real> v = res1[index];
        CudaVec4<real> v2 = b11[index];
        v.x += v2.x*f11;
        v.y += v2.y*f11;
        v.z += v2.z*f11;
        v2 = b12[index];
        v.x += v2.x*f12;
        v.y += v2.y*f12;
        v.z += v2.z*f12;
        v2 = b13[index];
        v.x += v2.x*f13;
        v.y += v2.y*f13;
        v.z += v2.z*f13;
        v2 = b14[index];
        v.x += v2.x*f14;
        v.y += v2.y*f14;
        v.z += v2.z*f14;
        res1[index] = v;
        v = res2[index];
        v2 = b21[index];
        v.x += v2.x*f21;
        v.y += v2.y*f21;
        v.z += v2.z*f21;
        v2 = b22[index];
        v.x += v2.x*f22;
        v.y += v2.y*f22;
        v.z += v2.z*f22;
        v2 = b23[index];
        v.x += v2.x*f23;
        v.y += v2.y*f23;
        v.z += v2.z*f23;
        v2 = b24[index];
        v.x += v2.x*f24;
        v.y += v2.y*f24;
        v.z += v2.z*f24;
        res2[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vAdd_kernel(int size, real* res, const real* a, const real* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vAdd_kernel(int size, real* res, const real* a, const real* b)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        //CudaVec3<real> ai = a[index];
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = ai + bi;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vAdd_kernel(int size, real* res, const real* a, const real* b)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        //CudaVec3<real> ai = a[index];
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = ai + bi;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vAdd_kernel(int size, CudaVec4<real>* res, const CudaVec4<real>* a, const CudaVec4<real>* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index];
        CudaVec4<real> v = a[index];
        CudaVec4<real> v2 = b[index];
        v.x += v2.x;
        v.y += v2.y;
        v.z += v2.z;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vAddCoord_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* a, const CudaRigidCoord3<real>* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        CudaRigidCoord3<real> v = a[index];
        CudaRigidCoord3<real> v2 = b[index];
        res[index] = v + v2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vAddCoordDeriv_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* a, const CudaRigidDeriv3<real>* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    {
        CudaRigidCoord3<real> v = a[index];
        CudaRigidDeriv3<real> v2 = b[index];

        res[index] = v + v2;

    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vAddDeriv_kernel(int size, real* res, const real* a, const real* b)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vOp_kernel(int size, real* res, const real* a, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vOp_kernel(int size, real* res, const real* a, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        //CudaVec3<real> ai = a[index];
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = ai + bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vOp_kernel(int size, real* res, const real* a, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        //CudaVec3<real> ai = a[index];
        //CudaVec3<real> bi = b[index];
        //CudaVec3<real> ri = ai + bi * f;
        //res[index] = ri;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vOp_kernel(int size, CudaVec4<real>* res, const CudaVec4<real>* a, const CudaVec4<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        CudaVec4<real> v = a[index];
        CudaVec4<real> v2 = b[index];
        v.x += v2.x*f;
        v.y += v2.y*f;
        v.z += v2.z*f;
        res[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vOpCoord_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* a, const CudaRigidCoord3<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        CudaRigidCoord3<real> v = a[index];
        CudaRigidCoord3<real> v2 = b[index];

        // following RigidTypes : only multiplying the position of the center ?
        v2.pos[0] *= f;
        v2.pos[1] *= f;
        v2.pos[2] *= f;

        res[index] = v + v2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vOpCoordDeriv_kernel(int size, CudaRigidCoord3<real>* res, const CudaRigidCoord3<real>* a, const CudaRigidDeriv3<real>* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    {
        CudaRigidCoord3<real> v = a[index];
        CudaRigidDeriv3<real> v2 = b[index];

        // following RigidTypes : multiplying everything ?
        v2.pos.x *= f;
        v2.pos.y *= f;
        v2.pos.z *= f;

        v2.rot.x *= f;
        v2.rot.y *= f;
        v2.rot.z *= f;

        res[index] = v + v2;

    }
}

template<class real>
__global__ void MechanicalObjectCudaRigid3t_vOpDeriv_kernel(int size, real* res, const real* a, const real* b, real f)
{
    int index = umul24(blockIdx.x,BSIZE*6)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vOp2_kernel(int size, real* res1, const real* a1, const real* b1, real f1, real* res2, const real* a2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vOp2_kernel(int size, real* res1, const real* a1, const real* b1, real f1, real* res2, const real* a2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vOp2_kernel(int size, real* res1, const real* a1, const real* b1, real f1, real* res2, const real* a2, const real* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vOp2_kernel(int size, CudaVec4<real>* res1, const CudaVec4<real>* a1, const CudaVec4<real>* b1, real f1, CudaVec4<real>* res2, const CudaVec4<real>* a2, const CudaVec4<real>* b2, real f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        CudaVec4<real> v = a1[index];
        CudaVec4<real> v2 = b1[index];
        v.x += v2.x*f1;
        v.y += v2.y*f1;
        v.z += v2.z*f1;
        res1[index] = v;
        v = a2[index];
        v2 = b2[index];
        v.x += v2.x*f2;
        v.y += v2.y*f2;
        v.z += v2.z*f2;
        res2[index] = v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vIntegrate_kernel(int size, const real* a, real* v, real* x, real f_v_v, real f_v_a, real f_x_x, real f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        real vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vIntegrate_kernel(int size, const real* a, real* v, real* x, real f_v_v, real f_v_a, real f_x_x, real f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        real vi;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vIntegrate_kernel(int size, const real* a, real* v, real* x, real f_v_v, real f_v_a, real f_x_x, real f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        real vi;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t1_vIntegrate_kernel(int size, const CudaVec4<real>* a, CudaVec4<real>* v, CudaVec4<real>* x, real f_v_v, real f_v_a, real f_x_x, real f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        CudaVec4<real> ai = a[index];
        CudaVec4<real> vi = v[index];
        CudaVec4<real> xi = x[index];
        vi.x = vi.x*f_v_v + ai.x*f_v_a;
        vi.y = vi.y*f_v_v + ai.y*f_v_a;
        vi.z = vi.z*f_v_v + ai.z*f_v_a;
        xi.x = xi.x*f_x_x + vi.x*f_x_v;
        xi.y = xi.y*f_x_x + vi.y*f_x_v;
        xi.z = xi.z*f_x_x + vi.z*f_x_v;
        v[index] = vi;
        x[index] = xi;
    }
}

#define RED_BSIZE 128
#define blockSize RED_BSIZE
//template<unsigned int blockSize>
__global__ void MechanicalObjectCudaVec_vDot_kernel(unsigned int n, float* res, const float* a, const float* b)
{
    extern __shared__ float fdata[];
//    __shared__ float fdata[blockSize];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = gridDim.x*(blockSize);
    fdata[tid] = 0;
    while (i < n) { fdata[tid] += a[i] * b[i]; i += gridSize; }
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { fdata[tid] += fdata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { fdata[tid] += fdata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { fdata[tid] += fdata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32)
    {
        volatile float * smem = fdata;
#if blockSize >= 64
        //if (blockSize >= 64)
        smem[tid] += smem[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
        smem[tid] += smem[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
        smem[tid] += smem[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
        smem[tid] += smem[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
        smem[tid] += smem[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
        smem[tid] += smem[tid + 1];
#endif
    }
    if (tid == 0) res[blockIdx.x] = fdata[0];
}

__global__ void MechanicalObjectCudaVec_vDot_kernel(unsigned int n, double* res, const double* a, const double* b)
{
    extern __shared__ double ddata[];
//    __shared__ double ddata[blockSize];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = gridDim.x*(blockSize);
    ddata[tid] = 0;
    while (i < n) { ddata[tid] += a[i] * b[i]; i += gridSize; }
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { ddata[tid] += ddata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { ddata[tid] += ddata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { ddata[tid] += ddata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32)
    {
        volatile double * smem = ddata;
#if blockSize >= 64
        //if (blockSize >= 64)
        smem[tid] += smem[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
        smem[tid] += smem[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
        smem[tid] += smem[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
        smem[tid] += smem[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
        smem[tid] += smem[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
        smem[tid] += smem[tid + 1];
#endif
    }
    if (tid == 0) res[blockIdx.x] = ddata[0];
}

//template<unsigned int blockSize>
template<class real>
__global__ void MechanicalObjectCudaVec_vSum_kernel(int n, real* res, const real* a)
{
    extern __shared__ real sdata[];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += a[i]; i += gridSize; }
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32)
    {
        volatile real * smem = sdata;
#if blockSize >= 64
        //if (blockSize >= 64)
        smem[tid] += smem[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
        smem[tid] += smem[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
        smem[tid] += smem[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
        smem[tid] += smem[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
        smem[tid] += smem[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
        smem[tid] += smem[tid + 1];
#endif
    }
    if (tid == 0) res[blockIdx.x] = sdata[0];
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vMultiOpA1B3_kernel(int size, real* res1, const real* a1, real* res2, const real* a21, real f21, const real* a22, real f22, const real* a23, real f23)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vMultiOpA1B3_kernel(int size, real* res1, const real* a1, real* res2, const real* a21, real f21, const real* a22, real f22, const real* a23, real f23)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
        index += BSIZE;
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vMultiOpA1B3_kernel(int size, real* res1, const real* a1, real* res2, const real* a21, real f21, const real* a22, real f22, const real* a23, real f23)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
        index += BSIZE;
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
        index += BSIZE;
        res1[index] = a1[index];
        res2[index] = a21[index] * f21 + a22[index] * f22 + a23[index] * f23;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vOp4_kernel(int size, real* res1, const real* a11, real f11, const real* a12, real f12, const real* a13, real f13, const real* a14, real f14)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vOp4_kernel(int size, real* res1, const real* a11, real f11, const real* a12, real f12, const real* a13, real f13, const real* a14, real f14)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
        index += BSIZE;
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vOp4_kernel(int size, real* res1, const real* a11, real f11, const real* a12, real f12, const real* a13, real f13, const real* a14, real f14)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
        index += BSIZE;
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
        index += BSIZE;
        res1[index] = a11[index] * f11 + a12[index] * f12 + a13[index] * f13 + a14[index] * f14;
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec1t_vOpMCNAB_kernel(int size, real* res1, const real* a11, const real* a12, real f12, real* res2, const real* a21, real* res3, const real* a31)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec2t_vOpMCNAB_kernel(int size, real* res1, const real* a11, const real* a12, real f12, real* res2, const real* a21, real* res3, const real* a31)
{
    int index = umul24(blockIdx.x,BSIZE*2)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
        index += BSIZE;
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
    }
}

template<class real>
__global__ void MechanicalObjectCudaVec3t_vOpMCNAB_kernel(int size, real* res1, const real* a11, const real* a12, real f12, real* res2, const real* a21, real* res3, const real* a31)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
        index += BSIZE;
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
        index += BSIZE;
        res1[index] = a11[index] + a12[index] * f12;
        res2[index] = a21[index];
        res3[index] = a31[index];
    }
}





//////////////////////
// CPU-side methods //
//////////////////////

void MechanicalObjectCudaVec1f_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAssign_kernel<float><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec2f_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec2t_vAssign_kernel<float><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*2*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t_vAssign_kernel<float><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*3*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t1_vAssign_kernel<float><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*4*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec6f_vAssign(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*6*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaRigid3f_vAssignCoord(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*7*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaRigid3f_vAssignDeriv(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*6*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec1f_vClear(unsigned int size, void* res)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec2t_vClear_kernel<float><<< grid, threads >>>(size, (CudaVec3<real>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<float><<< grid, threads >>>(3*size, (float*)res);
    cudaMemset(res, 0, size*sizeof(float));
}

void MechanicalObjectCudaVec2f_vClear(unsigned int size, void* res)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec2t_vClear_kernel<float><<< grid, threads >>>(size, (CudaVec3<real>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<float><<< grid, threads >>>(3*size, (float*)res);
    cudaMemset(res, 0, size*2*sizeof(float));
}

void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t_vClear_kernel<float><<< grid, threads >>>(size, (CudaVec3<real>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<float><<< grid, threads >>>(3*size, (float*)res);
    cudaMemset(res, 0, size*3*sizeof(float));
}

void MechanicalObjectCudaVec3f1_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t1_vClear_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<float><<< grid, threads >>>(4*size, (float*)res);
    cudaMemset(res, 0, size*4*sizeof(float));
}

void MechanicalObjectCudaVec6f_vClear(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*6*sizeof(float));
}

void MechanicalObjectCudaRigid3f_vClearCoord(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*7*sizeof(float));
}

void MechanicalObjectCudaRigid3f_vClearDeriv(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*6*sizeof(float));
}

void MechanicalObjectCudaVec1f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vMEq_kernel<float><<< grid, threads >>>(size, (float*)res, f); mycudaDebugError("MechanicalObjectCudaVec1t_vMEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, f);
}

void MechanicalObjectCudaVec2f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vMEq_kernel<float><<< grid, threads >>>(size, (float*)res, f); mycudaDebugError("MechanicalObjectCudaVec2t_vMEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, f);
}

void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vMEq_kernel<float><<< grid, threads >>>(size, (float*)res, f); mycudaDebugError("MechanicalObjectCudaVec3t_vMEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, f);
}

void MechanicalObjectCudaVec3f1_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vMEq_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vMEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<float><<< grid, threads >>>(4*size, (float*)res, f);
}

void MechanicalObjectCudaVec6f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vMEqCoord(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vMEqDeriv(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec1t_vEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec2f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vEqBF_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, (const CudaVec4<float>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vEqBF_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<float><<< grid, threads >>>(4*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec6f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vEqBFCoord(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vEqBFDeriv(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vPEq_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a); mycudaDebugError("MechanicalObjectCudaVec1t_vPEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec2f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEq_kernel<float><<< grid, threads >>>(size, (CudaVec2<float>*)res, (const CudaVec2<float>*)a); mycudaDebugError("MechanicalObjectCudaVec2t_vPEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEq_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a); mycudaDebugError("MechanicalObjectCudaVec3t_vPEq_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEq_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, (const CudaVec4<float>*)a); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEq_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<float><<< grid, threads >>>(4*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec6f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqCoord(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqCoordDeriv(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqCoordDeriv_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidDeriv3<float>*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqCoordDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqDeriv(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vPEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec1t_vPEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec2f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vPEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEqBF_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vPEqBF_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEqBF_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, (const CudaVec4<float>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEqBF_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<float><<< grid, threads >>>(4*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec6f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqBFCoord(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFCoordDeriv_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidDeriv3<float>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFCoordDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vPEqBFDeriv(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vPEqBF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b1, f1, (float*)res2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec1t_vPEqBF2_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEqBF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b1, f1, (float*)res2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec2t_vPEqBF2_kernel<float>");}
}

void MechanicalObjectCudaVec3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEqBF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b1, f1, (float*)res2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t_vPEqBF2_kernel<float>");}
}

void MechanicalObjectCudaVec3f1_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEqBF2_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res1, (const CudaVec4<float>*)b1, f1, (CudaVec4<float>*)res2, (const CudaVec4<float>*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEqBF2_kernel<float>");}
}

// void MechanicalObjectCudaVec6f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaVec6t_vPEqBF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b1, f1, (float*)res2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec6t_vPEqBF2_kernel<float>");}
// }

// void MechanicalObjectCudaRigid3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
// {
//   dim3 threads(BSIZE,1);
//   dim3 grid((size+BSIZE-1)/BSIZE,1);
//   {MechanicalObjectCudaRigid3t_vPEqBF2_kernel<float><<< grid, threads >>>(size, (CudaRigid3<float>*)res1, (const CudaRigid3<float>*)b1, f1, (CudaRigid3<float>*)res2, (const CudaRigid3<float>*)b2, f2); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBF2_kernel<float>");}
// }

void MechanicalObjectCudaVec1f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
            (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24); mycudaDebugError("MechanicalObjectCudaVec1t_vPEq4BF2_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec2t_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
            (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24);
}

void MechanicalObjectCudaVec3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3t_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
            (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24);
}

void MechanicalObjectCudaVec3f1_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3t1_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res1, (const CudaVec4<float>*)b11, f11, (const CudaVec4<float>*)b12, f12, (const CudaVec4<float>*)b13, f13, (const CudaVec4<float>*)b14, f14,
            (CudaVec4<float>*)res2, (const CudaVec4<float>*)b21, f21, (const CudaVec4<float>*)b22, f22, (const CudaVec4<float>*)b23, f23, (const CudaVec4<float>*)b24, f24);
}

// void MechanicalObjectCudaVec6f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
//                                                            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     MechanicalObjectCudaVec6t_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
//                                                                          (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24);
// }

// void MechanicalObjectCudaRigid3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
//                                                            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     MechanicalObjectCudaRigid3t_vPEq4BF2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
//                                                                          (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24);
// }

void MechanicalObjectCudaVec1f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vOp2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec1t_vOp2_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec2t_vOp2_kernel<float>");}
}

void MechanicalObjectCudaVec3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t_vOp2_kernel<float>");}
}

void MechanicalObjectCudaVec3f1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vOp2_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res1, (const CudaVec4<float>*)a1, (const CudaVec4<float>*)b1, f1, (CudaVec4<float>*)res2, (const CudaVec4<float>*)a2, (const CudaVec4<float>*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t1_vOp2_kernel<float>");}
}

// void MechanicalObjectCudaVec6f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     {MechanicalObjectCudaVec6t_vOp2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec6t_vOp2_kernel<float>");}
// }

// void MechanicalObjectCudaRigid3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     {MechanicalObjectCudaRigid3t_vOp2_kernel<float><<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2); mycudaDebugError("MechanicalObjectCudaRigid3t_vOp2_kernel<float>");}
// }

void MechanicalObjectCudaVec1f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vAdd_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b); mycudaDebugError("MechanicalObjectCudaVec1t_vAdd_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b);
}


void MechanicalObjectCudaVec2f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vAdd_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b); mycudaDebugError("MechanicalObjectCudaVec2t_vAdd_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vAdd_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b); mycudaDebugError("MechanicalObjectCudaVec3t_vAdd_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec3f1_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vAdd_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, (const CudaVec4<float>*)a, (const CudaVec4<float>*)b); mycudaDebugError("MechanicalObjectCudaVec3t1_vAdd_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<float><<< grid, threads >>>(4*size, (float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec6f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vAddCoord(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)a, (const CudaRigidCoord3<float>*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddCoordDeriv_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)a, (const CudaRigidDeriv3<float>*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddCoordDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vAddDeriv(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vOp_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec1t_vOp_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b, f);
}

void MechanicalObjectCudaVec2f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vOp_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vOp_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<float><<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vOp_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)res, (const CudaVec4<float>*)a, (const CudaVec4<float>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vOp_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<float><<< grid, threads >>>(4*size, (float*)res, (const float*)a, (const float*)b, f);
}

void MechanicalObjectCudaVec6f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vOpCoord(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpCoord_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)a, (const CudaRigidCoord3<float>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpCoord_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpCoordDeriv_kernel<float><<< grid, threads >>>(size, (CudaRigidCoord3<float>*)res, (const CudaRigidCoord3<float>*)a, (const CudaRigidDeriv3<float>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpCoordDeriv_kernel<float>");}
}

void MechanicalObjectCudaRigid3f_vOpDeriv(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpDeriv_kernel<float><<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpDeriv_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec1t_vIntegrate_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<float><<< grid, threads >>>(3*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec2f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec2t_vIntegrate_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<float><<< grid, threads >>>(3*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec3t_vIntegrate_kernel<float>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<float><<< grid, threads >>>(3*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec3f1_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vIntegrate_kernel<float><<< grid, threads >>>(size, (const CudaVec4<float>*)a, (CudaVec4<float>*)v, (CudaVec4<float>*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec3t1_vIntegrate_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<float><<< grid, threads >>>(4*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

// void MechanicalObjectCudaVec6f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaVec6t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec6t_vIntegrate_kernel<float>");}
// }

// void MechanicalObjectCudaRigid3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaRigid3t_vIntegrate_kernel<float><<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaRigid3t_vIntegrate_kernel<float>");}
// }

int MechanicalObjectCudaVec1f_vDotTmpSize(unsigned int size)
{
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

int MechanicalObjectCudaVec2f_vDotTmpSize(unsigned int size)
{
    size *= 2;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec1f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            mycudaDebugError("cudaMemcpy::MechanicalObjectCudaVec1f_vDot<float>");
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<float><<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<float>");}
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            mycudaDebugError("cudaMemcpy::MechanicalObjectCudaVec1f_vDot<float>");
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

void MechanicalObjectCudaVec2f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 2;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<float><<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<float>");}
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 3;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<float><<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<float>");}
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec3f1_vDotTmpSize(unsigned int size)
{
    size *= 4;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs; //(nblocs+3)/4;
}

void MechanicalObjectCudaVec3f1_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 4;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<float><<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<float>");}
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec6f_vDotTmpSize(unsigned int size)
{
    size *= 6;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec6f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 6;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<float><<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<float>");}
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaRigid3f_vDotTmpSize(unsigned int size)
{
    size *= 6;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaRigid3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 6;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<float>*/ <<< grid, threads , RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

void MechanicalObjectCudaVec1f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vMultiOpA1B3_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a1, (float*) res2, (const float*) a21, f21, (const float*) a22, f22, (const float*) a23, f23); mycudaDebugError("MechanicalObjectCudaVec1t_vMultiOpA1B3_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vMultiOpA1B3_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a1, (float*) res2, (const float*) a21, f21, (const float*) a22, f22, (const float*) a23, f23); mycudaDebugError("MechanicalObjectCudaVec2t_vMultiOpA1B3_kernel<float>");}
}

void MechanicalObjectCudaVec3f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vMultiOpA1B3_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a1, (float*) res2, (const float*) a21, f21, (const float*) a22, f22, (const float*) a23, f23); mycudaDebugError("MechanicalObjectCudaVec3t_vMultiOpA1B3_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vOp4_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, f11, (const float*) a12, f12, (const float*) a13, f13, (const float*) a14, f14); mycudaDebugError("MechanicalObjectCudaVec1t_vOp4_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp4_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, f11, (const float*) a12, f12, (const float*) a13, f13, (const float*) a14, f14); mycudaDebugError("MechanicalObjectCudaVec2t_vOp4_kernel<float>");}
}

void MechanicalObjectCudaVec3f_vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp4_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, f11, (const float*) a12, f12, (const float*) a13, f13, (const float*) a14, f14); mycudaDebugError("MechanicalObjectCudaVec3t_vOp4_kernel<float>");}
}

void MechanicalObjectCudaVec1f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec1t_vOpMCNAB_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, (const float*) a12, f12, (float*) res2, (const float*) a21, (float*) res3, (const float*) a31); mycudaDebugError("MechanicalObjectCudaVec1t_vOpMCNAB_kernel<float>");}
}

void MechanicalObjectCudaVec2f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOpMCNAB_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, (const float*) a12, f12, (float*) res2, (const float*) a21, (float*) res3, (const float*) a31); mycudaDebugError("MechanicalObjectCudaVec2t_vOpMCNAB_kernel<float>");}
}

void MechanicalObjectCudaVec3f_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOpMCNAB_kernel<float><<< grid, threads >>>(size, (float*) res1, (const float*) a11, (const float*) a12, f12, (float*) res2, (const float*) a21, (float*) res3, (const float*) a31); mycudaDebugError("MechanicalObjectCudaVec3t_vOpMCNAB_kernel<float>");}
}



#ifdef SOFA_GPU_CUDA_DOUBLE

void MechanicalObjectCudaVec2d_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec2t_vAssign_kernel<double><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*2*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3d_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t_vAssign_kernel<double><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*3*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3d1_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t1_vAssign_kernel<double><<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*4*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec6d_vAssign(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*6*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaRigid3d_vAssignCoord(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*7*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaRigid3d_vAssignDeriv(unsigned int size, void* res, const void* a)
{
    cudaMemcpy(res, a, size*6*sizeof(double), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec2d_vClear(unsigned int size, void* res)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec2t_vClear_kernel<double><<< grid, threads >>>(size, (CudaVec3<real>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<double><<< grid, threads >>>(3*size, (double*)res);
    cudaMemset(res, 0, size*2*sizeof(double));
}

void MechanicalObjectCudaVec3d_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t_vClear_kernel<double><<< grid, threads >>>(size, (CudaVec3<real>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<double><<< grid, threads >>>(3*size, (double*)res);
    cudaMemset(res, 0, size*3*sizeof(double));
}

void MechanicalObjectCudaVec3d1_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3t1_vClear_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vClear_kernel<double><<< grid, threads >>>(4*size, (double*)res);
    cudaMemset(res, 0, size*4*sizeof(double));
}

void MechanicalObjectCudaVec6d_vClear(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*6*sizeof(double));
}

void MechanicalObjectCudaRigid3d_vClearCoord(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*7*sizeof(double));
}

void MechanicalObjectCudaRigid3d_vClearDeriv(unsigned int size, void* res)
{
    cudaMemset(res, 0, size*6*sizeof(double));
}

void MechanicalObjectCudaVec2d_vMEq(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vMEq_kernel<double><<< grid, threads >>>(size, (double*)res, f); mycudaDebugError("MechanicalObjectCudaVec2t_vMEq_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<double><<< grid, threads >>>(3*size, (double*)res, f);
}

void MechanicalObjectCudaVec3d_vMEq(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vMEq_kernel<double><<< grid, threads >>>(size, (double*)res, f); mycudaDebugError("MechanicalObjectCudaVec3t_vMEq_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<double><<< grid, threads >>>(3*size, (double*)res, f);
}

void MechanicalObjectCudaVec3d1_vMEq(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vMEq_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vMEq_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vMEq_kernel<double><<< grid, threads >>>(4*size, (double*)res, f);
}

void MechanicalObjectCudaVec6d_vMEq(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vMEqCoord(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vMEqDeriv(unsigned int size, void* res, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vMEqDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2f_vEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vEqBF_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec3d_vEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vEqBF_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec3d1_vEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vEqBF_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, (const CudaVec4<double>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vEqBF_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vEqBF_kernel<double><<< grid, threads >>>(4*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec6d_vEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vEqBFCoord(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vEqBFDeriv(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vEqBFDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEq_kernel<double><<< grid, threads >>>(size, (CudaVec2<double>*)res, (const CudaVec2<double>*)a); mycudaDebugError("MechanicalObjectCudaVec2t_vPEq_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a);
}

void MechanicalObjectCudaVec3d_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEq_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a); mycudaDebugError("MechanicalObjectCudaVec3t_vPEq_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a);
}

void MechanicalObjectCudaVec3d1_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEq_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, (const CudaVec4<double>*)a); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEq_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEq_kernel<double><<< grid, threads >>>(4*size, (double*)res, (const double*)a);
}

void MechanicalObjectCudaVec6d_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqCoord(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqCoordDeriv(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqCoordDeriv_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidDeriv3<double>*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqCoordDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqDeriv(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vPEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vPEqBF_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec3d_vPEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEqBF_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vPEqBF_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec3d1_vPEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEqBF_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, (const CudaVec4<double>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEqBF_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vPEqBF_kernel<double><<< grid, threads >>>(4*size, (double*)res, (const double*)b, f);
}

void MechanicalObjectCudaVec6d_vPEqBF(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqBFCoord(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFCoordDeriv_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidDeriv3<double>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFCoordDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vPEqBFDeriv(unsigned int size, void* res, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBFDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2f_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vPEqBF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b1, f1, (double*)res2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec2t_vPEqBF2_kernel<double>");}
}

void MechanicalObjectCudaVec3d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vPEqBF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b1, f1, (double*)res2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t_vPEqBF2_kernel<double>");}
}

void MechanicalObjectCudaVec3d1_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vPEqBF2_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res1, (const CudaVec4<double>*)b1, f1, (CudaVec4<double>*)res2, (const CudaVec4<double>*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t1_vPEqBF2_kernel<double>");}
}

// void MechanicalObjectCudaVec6d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaVec6t_vPEqBF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b1, f1, (double*)res2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec6t_vPEqBF2_kernel<double>");}
// }

// void MechanicalObjectCudaRigid3d_vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
// {
//   dim3 threads(BSIZE,1);
//   dim3 grid((size+BSIZE-1)/BSIZE,1);
//   {MechanicalObjectCudaRigid3t_vPEqBF2_kernel<double><<< grid, threads >>>(size, (CudaRigid3<double>*)res1, (const CudaRigid3<double>*)b1, f1, (CudaRigid3<double>*)res2, (const CudaRigid3<double>*)b2, f2); mycudaDebugError("MechanicalObjectCudaRigid3t_vPEqBF2_kernel<double>");}
// }

void MechanicalObjectCudaVec2d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
        void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec2t_vPEq4BF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b11, f11, (const double*)b12, f12, (const double*)b13, f13, (const double*)b14, f14,
            (double*)res2, (const double*)b21, f21, (const double*)b22, f22, (const double*)b23, f23, (const double*)b24, f24);
}

void MechanicalObjectCudaVec3d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
        void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3t_vPEq4BF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b11, f11, (const double*)b12, f12, (const double*)b13, f13, (const double*)b14, f14,
            (double*)res2, (const double*)b21, f21, (const double*)b22, f22, (const double*)b23, f23, (const double*)b24, f24);
}

void MechanicalObjectCudaVec3d1_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
        void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3t1_vPEq4BF2_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res1, (const CudaVec4<double>*)b11, f11, (const CudaVec4<double>*)b12, f12, (const CudaVec4<double>*)b13, f13, (const CudaVec4<double>*)b14, f14,
            (CudaVec4<double>*)res2, (const CudaVec4<double>*)b21, f21, (const CudaVec4<double>*)b22, f22, (const CudaVec4<double>*)b23, f23, (const CudaVec4<double>*)b24, f24);
}

// void MechanicalObjectCudaVec6d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
//                                                            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     MechanicalObjectCudaVec6t_vPEq4BF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b11, f11, (const double*)b12, f12, (const double*)b13, f13, (const double*)b14, f14,
//                                                                          (double*)res2, (const double*)b21, f21, (const double*)b22, f22, (const double*)b23, f23, (const double*)b24, f24);
// }

// void MechanicalObjectCudaRigid3d_vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
//                                                            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     MechanicalObjectCudaRigid3t_vPEq4BF2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)b11, f11, (const double*)b12, f12, (const double*)b13, f13, (const double*)b14, f14,
//                                                                          (double*)res2, (const double*)b21, f21, (const double*)b22, f22, (const double*)b23, f23, (const double*)b24, f24);
// }

void MechanicalObjectCudaVec2d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)a1, (const double*)b1, f1, (double*)res2, (const double*)a2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec2t_vOp2_kernel<double>");}
}

void MechanicalObjectCudaVec3d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)a1, (const double*)b1, f1, (double*)res2, (const double*)a2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t_vOp2_kernel<double>");}
}

void MechanicalObjectCudaVec3d1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vOp2_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res1, (const CudaVec4<double>*)a1, (const CudaVec4<double>*)b1, f1, (CudaVec4<double>*)res2, (const CudaVec4<double>*)a2, (const CudaVec4<double>*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t1_vOp2_kernel<double>");}
}

// void MechanicalObjectCudaVec6d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     {MechanicalObjectCudaVec6t_vOp2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)a1, (const double*)b1, f1, (double*)res2, (const double*)a2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec6t_vOp2_kernel<double>");}
// }

// void MechanicalObjectCudaRigid3d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1,
//     double f1, void* res2, const void* a2, const void* b2, double f2)
// {
//     dim3 threads(BSIZE,1);
//     dim3 grid((size+BSIZE-1)/BSIZE,1);
//     {MechanicalObjectCudaVec3t_vOp2_kernel<double><<< grid, threads >>>(size, (double*)res1, (const double*)a1, (const double*)b1, f1, (double*)res2, (const double*)a2, (const double*)b2, f2); mycudaDebugError("MechanicalObjectCudaVec3t_vOp2_kernel<double>");}
// }

void MechanicalObjectCudaVec2d_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vAdd_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b); mycudaDebugError("MechanicalObjectCudaVec2t_vAdd_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a, (const double*)b);
}

void MechanicalObjectCudaVec3d_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vAdd_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b); mycudaDebugError("MechanicalObjectCudaVec3t_vAdd_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a, (const double*)b);
}

void MechanicalObjectCudaVec3d1_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vAdd_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, (const CudaVec4<double>*)a, (const CudaVec4<double>*)b); mycudaDebugError("MechanicalObjectCudaVec3t1_vAdd_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vAdd_kernel<double><<< grid, threads >>>(4*size, (double*)res, (const double*)a, (const double*)b);
}

void MechanicalObjectCudaVec6d_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vAddCoord(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)a, (const CudaRigidCoord3<double>*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddCoordDeriv_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)a, (const CudaRigidDeriv3<double>*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddCoordDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vAddDeriv(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vAddDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b); mycudaDebugError("MechanicalObjectCudaRigid3t_vAddDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vOp(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec2t_vOp_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a, (const double*)b, f);
}

void MechanicalObjectCudaVec3d_vOp(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t_vOp_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<double><<< grid, threads >>>(3*size, (double*)res, (const double*)a, (const double*)b, f);
}

void MechanicalObjectCudaVec3d1_vOp(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vOp_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)res, (const CudaVec4<double>*)a, (const CudaVec4<double>*)b, f); mycudaDebugError("MechanicalObjectCudaVec3t1_vOp_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vOp_kernel<double><<< grid, threads >>>(4*size, (double*)res, (const double*)a, (const double*)b, f);
}

void MechanicalObjectCudaVec6d_vOp(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vOpCoord(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpCoord_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)a, (const CudaRigidCoord3<double>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpCoord_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpCoordDeriv_kernel<double><<< grid, threads >>>(size, (CudaRigidCoord3<double>*)res, (const CudaRigidCoord3<double>*)a, (const CudaRigidDeriv3<double>*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpCoordDeriv_kernel<double>");}
}

void MechanicalObjectCudaRigid3d_vOpDeriv(unsigned int size, void* res, const void* a, const void* b, double f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaRigid3t_vOpDeriv_kernel<double><<< grid, threads >>>(size, (double*)res, (const double*)a, (const double*)b, f); mycudaDebugError("MechanicalObjectCudaRigid3t_vOpDeriv_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vIntegrate_kernel<double><<< grid, threads >>>(size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec2t_vIntegrate_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<double><<< grid, threads >>>(3*size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vIntegrate_kernel<double><<< grid, threads >>>(size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec3t_vIntegrate_kernel<double>");}
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<double><<< grid, threads >>>(3*size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec3d1_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t1_vIntegrate_kernel<double><<< grid, threads >>>(size, (const CudaVec4<double>*)a, (CudaVec4<double>*)v, (CudaVec4<double>*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec3t1_vIntegrate_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1t_vIntegrate_kernel<double><<< grid, threads >>>(4*size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

// void MechanicalObjectCudaVec6d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaVec6t_vIntegrate_kernel<double><<< grid, threads >>>(size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaVec6t_vIntegrate_kernel<double>");}
// }

// void MechanicalObjectCudaRigid3d_vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
// {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	{MechanicalObjectCudaRigid3t_vIntegrate_kernel<double><<< grid, threads >>>(size, (const double*)a, (double*)v, (double*)x, f_v_v, f_v_a, f_x_x, f_x_v); mycudaDebugError("MechanicalObjectCudaRigid3t_vIntegrate_kernel<double>");}
// }

int MechanicalObjectCudaVec2d_vDotTmpSize(unsigned int size)
{
    size *= 2;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec2d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 2;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<double><<< grid, threads, RED_BSIZE * sizeof(double) >>>(nblocs, (double*)tmp, (const double*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<double>");}
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec3d_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 3;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<double><<< grid, threads, RED_BSIZE * sizeof(double) >>>(nblocs, (double*)tmp, (const double*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<double>");}
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec3d1_vDotTmpSize(unsigned int size)
{
    size *= 4;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs; //(nblocs+3)/4;
}

void MechanicalObjectCudaVec3d1_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 4;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<double><<< grid, threads, RED_BSIZE * sizeof(double) >>>(nblocs, (double*)tmp, (const double*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<double>");}
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec6d_vDotTmpSize(unsigned int size)
{
    size *= 6;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaVec6d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 6;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            {MechanicalObjectCudaVec_vSum_kernel<double><<< grid, threads, RED_BSIZE * sizeof(double) >>>(nblocs, (double*)tmp, (const double*)tmp); mycudaDebugError("MechanicalObjectCudaVec_vSum_kernel<double>");}
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaRigid3d_vDotTmpSize(unsigned int size)
{
    size *= 6;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

void MechanicalObjectCudaRigid3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* rtmp)
{
    size *= 6;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVec_vDot_kernel /*<double>*/ <<< grid, threads , RED_BSIZE * sizeof(double) >>>(size, (double*)tmp, (const double*)a, (const double*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(double),cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(double),cudaMemcpyDeviceToHost);
            double r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

void MechanicalObjectCudaVec2d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vMultiOpA1B3_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a1, (double*) res2, (const double*) a21, f21, (const double*) a22, f22, (const double*) a23, f23); mycudaDebugError("MechanicalObjectCudaVec2t_vMultiOpA1B3_kernel<double>");}
}

void MechanicalObjectCudaVec3d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vMultiOpA1B3_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a1, (double*) res2, (const double*) a21, f21, (const double*) a22, f22, (const double*) a23, f23); mycudaDebugError("MechanicalObjectCudaVec3t_vMultiOpA1B3_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOp4_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a11, f11, (const double*) a12, f12, (const double*) a13, f13, (const double*) a14, f14); mycudaDebugError("MechanicalObjectCudaVec2t_vOp4_kernel<double>");}
}

void MechanicalObjectCudaVec3d_vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOp4_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a11, f11, (const double*) a12, f12, (const double*) a13, f13, (const double*) a14, f14); mycudaDebugError("MechanicalObjectCudaVec3t_vOp4_kernel<double>");}
}

void MechanicalObjectCudaVec2d_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec2t_vOpMCNAB_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a11, (const double*) a12, f12, (double*) res2, (const double*) a21, (double*) res3, (const double*) a31); mycudaDebugError("MechanicalObjectCudaVec2t_vOpMCNAB_kernel<double>");}
}

void MechanicalObjectCudaVec3d_vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MechanicalObjectCudaVec3t_vOpMCNAB_kernel<double><<< grid, threads >>>(size, (double*) res1, (const double*) a11, (const double*) a12, f12, (double*) res2, (const double*) a21, (double*) res3, (const double*) a31); mycudaDebugError("MechanicalObjectCudaVec3t_vOpMCNAB_kernel<double>");}
}


#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
