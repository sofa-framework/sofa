/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_INL

#include "CudaMechanicalObject.h"
#include <SofaBaseMechanics/MechanicalObject.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

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
//        void MechanicalObjectCudaVec3f1_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);


    void MechanicalObjectCudaVec6f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec6f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec6f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec6f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec6f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec6f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec6f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
//        void MechanicalObjectCudaVec6f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
//        void MechanicalObjectCudaVec6f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
//        void MechanicalObjectCudaVec6f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
//                void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
//        void MechanicalObjectCudaVec6f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaVec6f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec6f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
//        void MechanicalObjectCudaVec6f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);


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
    // void MechanicalObjectCudaRigid3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    //void MechanicalObjectCudaRigid3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
    //                                                            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    // void MechanicalObjectCudaRigid3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaRigid3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaRigid3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
//        void MechanicalObjectCudaRigid3f_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23);



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
//        void MechanicalObjectCudaVec3d1_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);


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
//        void MechanicalObjectCudaVec6d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);


    void MechanicalObjectCudaRigid3d_vAssignCoord(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vAssignDeriv(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaRigid3d_vAssign(unsigned int size, void* res, const void* a);
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
    // void MechanicalObjectCudaRigid3d_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    // void MechanicalObjectCudaRigid3d_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
    //                                                             void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    // void MechanicalObjectCudaRigid3d_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
    int MechanicalObjectCudaRigid3d_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaRigid3d_vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp);
//        void MechanicalObjectCudaRigid3d_vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"





template<>
class CudaKernelsMechanicalObject<CudaVec1fTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec1f_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec1f_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaVec1f_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec1f_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec1f_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec1f_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec1f_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaVec1f_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectCudaVec1f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
    {   MechanicalObjectCudaVec1f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
    {
        MechanicalObjectCudaVec1f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
    {   MechanicalObjectCudaVec1f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec1f_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaVec1f_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
    {   MechanicalObjectCudaVec1f_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23); }
    static void vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
    {   MechanicalObjectCudaVec1f_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14); }
    static void vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
    {   MechanicalObjectCudaVec1f_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31); }
};



template<>
class CudaKernelsMechanicalObject<CudaVec2fTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec2f_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec2f_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaVec2f_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec2f_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec2f_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec2f_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec2f_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaVec2f_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
    {   MechanicalObjectCudaVec2f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
    {   MechanicalObjectCudaVec2f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
    {
        MechanicalObjectCudaVec2f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
    {   MechanicalObjectCudaVec2f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec2f_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaVec2f_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
    {   MechanicalObjectCudaVec2f_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23); }
    static void vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
    {   MechanicalObjectCudaVec2f_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14); }
    static void vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
    {   MechanicalObjectCudaVec2f_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31); }
};

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
    static void vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, float f21, const void* a22, float f22, const void* a23, float f23)
    {   MechanicalObjectCudaVec3f_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23); }
    static void vOp4(unsigned int size, void* res1, const void* a11, float f11, const void* a12, float f12, const void* a13, float f13, const void* a14, float f14)
    {   MechanicalObjectCudaVec3f_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14); }
    static void vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, float f12, void* res2, const void* a21, void* res3, const void* a31)
    {   MechanicalObjectCudaVec3f_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31); }
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
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, float /*f21*/, const void* /*a22*/, float /*f22*/, const void* /*a23*/, float /*f23*/)
    {   /*MechanicalObjectCudaVec3f1_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, float /*f11*/, const void* /*a12*/, float /*f12*/, const void* /*a13*/, float /*f13*/, const void* /*a14*/, float /*f14*/)
    {   /*MechanicalObjectCudaVec3f1_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, float /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {   /*MechanicalObjectCudaVec3f1_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
};

template<>
class CudaKernelsMechanicalObject<CudaVec6fTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec6f_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec6f_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaVec6f_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec6f_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec6f_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaVec6f_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec6f_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaVec6f_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int /* size */, const void* /* a */, void* /* v */, void* /* x */, float /* f_v_v */, float /* f_v_a */, float /* f_x_x */, float /* f_x_v */)
    {   /* MechanicalObjectCudaVec6f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); */ }
    static void vPEqBF2(unsigned int /* size */, void* /* res1 */, const void* /* b1 */, float /* f1 */, void* /* res2 */, const void* /* b2 */, float /* f2 */)
    {   /* MechanicalObjectCudaVec6f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); */ }
    static void vPEq4BF2(unsigned int /* size */, void* /* res1 */, const void* /* b11 */, float /* f11 */, const void* /* b12 */, float /* f12 */, const void* /* b13 */, float /* f13 */, const void* /* b14 */, float /* f14 */,
            void* /* res2 */, const void* /* b21 */, float /* f21 */, const void* /* b22 */, float /* f22 */, const void* /* b23 */, float /* f23 */, const void* /* b24 */, float /* f24 */)
    {
        /* MechanicalObjectCudaVec6f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
        res2, b21, f21, b22, f22, b23, f23, b24, f24); */
    }
    static void vOp2(unsigned int /* size */, void* /* res1 */, const void* /* a1 */, const void* /* b1 */, float /* f1 */, void* /* res2 */, const void* /* a2 */, const void* /* b2 */, float /* f2 */)
    {   /* MechanicalObjectCudaVec6f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); */ }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec6f_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaVec6f_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, float /*f21*/, const void* /*a22*/, float /*f22*/, const void* /*a23*/, float /*f23*/)
    {   /*MechanicalObjectCudaVec6f_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, float /*f11*/, const void* /*a12*/, float /*f12*/, const void* /*a13*/, float /*f13*/, const void* /*a14*/, float /*f14*/)
    {   /*MechanicalObjectCudaVec6f_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, float /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {   /*MechanicalObjectCudaVec6f_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
};

template<>
class CudaKernelsMechanicalObject<CudaRigid3fTypes>
{
public:
    static void vAssignCoord(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3f_vAssignCoord(size, res, a); }
    static void vAssignDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3f_vAssignDeriv(size, res, a); }
    static void vClearCoord(unsigned int size, void* res)
    {   MechanicalObjectCudaRigid3f_vClearCoord(size, res); }
    static void vClearDeriv(unsigned int size, void* res)
    {   MechanicalObjectCudaRigid3f_vClearDeriv(size, res); }
    static void vMEqCoord(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaRigid3f_vMEqCoord(size, res, f); }
    static void vMEqDeriv(unsigned int size, void* res, float f)
    {   MechanicalObjectCudaRigid3f_vMEqDeriv(size, res, f); }
    static void vEqBFCoord(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vEqBFCoord(size, res, b, f); }
    static void vEqBFDeriv(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vEqBFDeriv(size, res, b, f); }
    static void vPEqCoord(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3f_vPEqCoord(size, res, a); }
    static void vPEqCoordDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3f_vPEqCoordDeriv(size, res, a); }
    static void vPEqDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3f_vPEqDeriv(size, res, a); }
    static void vPEqBFCoord(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vPEqBFCoord(size, res, b, f); }
    static void vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vPEqBFCoordDeriv(size, res, b, f); }
    static void vPEqBFDeriv(unsigned int size, void* res, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vPEqBFDeriv(size, res, b, f); }
    static void vAddCoord(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3f_vAddCoord(size, res, a, b); }
    static void vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3f_vAddCoordDeriv(size, res, a, b); }
    static void vAddDeriv(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3f_vAddDeriv(size, res, a, b); }
    static void vOpCoord(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vOpCoord(size, res, a, b, f); }
    static void vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vOpCoordDeriv(size, res, a, b, f); }
    static void vOpDeriv(unsigned int size, void* res, const void* a, const void* b, float f)
    {   MechanicalObjectCudaRigid3f_vOpDeriv(size, res, a, b, f); }
    static void vIntegrate(unsigned int /* size */, const void* /* a */, void* /* v */, void* /* x */, float /* f_v_v */, float /* f_v_a */, float /* f_x_x */, float /* f_x_v */)
    {  /* MechanicalObjectCudaRigid3f_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); */ }
    static void vPEqBF2(unsigned int /* size */, void* /* res1 */, const void* /* b1 */, float /* f1 */, void* /* res2 */, const void* /* b2 */, float /* f2 */)
    { /*  MechanicalObjectCudaRigid3f_vPEqBF2(size, res1, b1, f1, res2, b2, f2); */ }
    static void vPEq4BF2(unsigned int /* size */, void* /* res1 */, const void* /* b11 */, float /* f11 */, const void* /* b12 */, float /* f12 */, const void* /* b13 */, float /* f13 */, const void* /* b14 */, float /* f14 */,
            void* /* res2 */, const void* /* b21 */, float /* f21 */, const void* /* b22 */, float /* f22 */, const void* /* b23 */, float /* f23 */, const void* /* b24 */, float /* f24 */)
    {
        /* MechanicalObjectCudaRigid3f_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                                                  res2, b21, f21, b22, f22, b23, f23, b24, f24); */
    }
    static void vOp2(unsigned int /* size */, void* /* res1 */, const void* /* a1 */, const void* /* b1 */, float /* f1 */, void* /* res2 */, const void* /* a2 */, const void* /* b2 */, float /* f2 */)
    { /*  MechanicalObjectCudaRigid3f_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); */ }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaRigid3f_vDotTmpSize(size); }
    static void vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp)
    {   MechanicalObjectCudaRigid3f_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, float /*f21*/, const void* /*a22*/, float /*f22*/, const void* /*a23*/, float /*f23*/)
    {   /*MechanicalObjectCudaRigid3f_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, float /*f11*/, const void* /*a12*/, float /*f12*/, const void* /*a13*/, float /*f13*/, const void* /*a14*/, float /*f14*/)
    {   /*MechanicalObjectCudaRigid3f_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, float /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {   /*MechanicalObjectCudaRigid3f_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
};

#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
class CudaKernelsMechanicalObject<CudaVec2dTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec2d_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec2d_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaVec2d_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec2d_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec2d_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec2d_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec2d_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaVec2d_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int size, const void* a, void* v, void* x, double f_v_v, double f_v_a, double f_x_x, double f_x_v)
    {   MechanicalObjectCudaVec2d_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); }
    static void vPEqBF2(unsigned int size, void* res1, const void* b1, double f1, void* res2, const void* b2, double f2)
    {   MechanicalObjectCudaVec2d_vPEqBF2(size, res1, b1, f1, res2, b2, f2); }
    static void vPEq4BF2(unsigned int size, void* res1, const void* b11, double f11, const void* b12, double f12, const void* b13, double f13, const void* b14, double f14,
            void* res2, const void* b21, double f21, const void* b22, double f22, const void* b23, double f23, const void* b24, double f24)
    {
        MechanicalObjectCudaVec2d_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                res2, b21, f21, b22, f22, b23, f23, b24, f24);
    }
    static void vOp2(unsigned int size, void* res1, const void* a1, const void* b1, double f1, void* res2, const void* a2, const void* b2, double f2)
    {   MechanicalObjectCudaVec2d_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec2d_vDotTmpSize(size); }
    static void vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp)
    {   MechanicalObjectCudaVec2d_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23)
    {   MechanicalObjectCudaVec2d_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23); }
    static void vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14)
    {   MechanicalObjectCudaVec2d_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14); }
    static void vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31)
    {   MechanicalObjectCudaVec2d_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31); }
};



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
    static void vMultiOpA1B3(unsigned int size, void* res1, const void* a1, void* res2, const void* a21, double f21, const void* a22, double f22, const void* a23, double f23)
    {   MechanicalObjectCudaVec3d_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23); }
    static void vOp4(unsigned int size, void* res1, const void* a11, double f11, const void* a12, double f12, const void* a13, double f13, const void* a14, double f14)
    {   MechanicalObjectCudaVec3d_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14); }
    static void vOpMCNAB(unsigned int size, void* res1, const void* a11, const void* a12, double f12, void* res2, const void* a21, void* res3, const void* a31)
    {   MechanicalObjectCudaVec3d_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31); }
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
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, double /*f21*/, const void* /*a22*/, double /*f22*/, const void* /*a23*/, double /*f23*/)
    {  /* MechanicalObjectCudaVec3d1_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, double /*f11*/, const void* /*a12*/, double /*f12*/, const void* /*a13*/, double /*f13*/, const void* /*a14*/, double /*f14*/)
    {  /* MechanicalObjectCudaVec3d1_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, double /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {  /* MechanicalObjectCudaVec3d1_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
};

template<>
class CudaKernelsMechanicalObject<CudaVec6dTypes>
{
public:
    static void vAssign(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec6d_vAssign(size, res, a); }
    static void vClear(unsigned int size, void* res)
    {   MechanicalObjectCudaVec6d_vClear(size, res); }
    static void vMEq(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaVec6d_vMEq(size, res, f); }
    static void vEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec6d_vEqBF(size, res, b, f); }
    static void vPEq(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaVec6d_vPEq(size, res, a); }
    static void vPEqBF(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaVec6d_vPEqBF(size, res, b, f); }
    static void vAdd(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaVec6d_vAdd(size, res, a, b); }
    static void vOp(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaVec6d_vOp(size, res, a, b, f); }
    static void vIntegrate(unsigned int /* size */, const void* /* a */, void* /* v */, void* /* x */, double /* f_v_v */, double /* f_v_a */, double /* f_x_x */, double /* f_x_v */)
    { /*  MechanicalObjectCudaVec6d_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); */ }
    static void vPEqBF2(unsigned int /* size */, void* /* res1 */, const void* /* b1 */, double /* f1 */, void* /* res2 */, const void* /* b2 */, double /* f2 */)
    { /* MechanicalObjectCudaVec6d_vPEqBF2(size, res1, b1, f1, res2, b2, f2); */ }
    static void vPEq4BF2(unsigned int /* size */, void* /* res1 */, const void* /* b11 */, double /* f11 */, const void* /* b12 */, double /* f12 */, const void* /* b13 */, double /* f13 */, const void* /* b14 */, double /* f14 */,
            void* /* res2 */, const void* /* b21 */, double /* f21 */, const void* /* b22 */, double /* f22 */, const void* /* b23 */, double /* f23 */, const void* /* b24 */, double /* f24 */)
    {
        /*   MechanicalObjectCudaVec6d_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                                                    res2, b21, f21, b22, f22, b23, f23, b24, f24); */
    }
    static void vOp2(unsigned int /* size */, void* /* res1 */, const void* /* a1 */, const void* /* b1 */, double /* f1 */, void* /* res2 */, const void* /* a2 */, const void* /* b2 */, double /* f2 */)
    { /*  MechanicalObjectCudaVec6d_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); */ }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaVec6d_vDotTmpSize(size); }
    static void vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp)
    {   MechanicalObjectCudaVec6d_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, double /*f21*/, const void* /*a22*/, double /*f22*/, const void* /*a23*/, double /*f23*/)
    {  /* MechanicalObjectCudaVec6d_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, double /*f11*/, const void* /*a12*/, double /*f12*/, const void* /*a13*/, double /*f13*/, const void* /*a14*/, double /*f14*/)
    {  /* MechanicalObjectCudaVec6d_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, double /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {  /* MechanicalObjectCudaVec6d_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
};

template<>
class CudaKernelsMechanicalObject<CudaRigid3dTypes>
{
public:
    static void vAssignCoord(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3d_vAssignCoord(size, res, a); }
    static void vAssignDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3d_vAssignDeriv(size, res, a); }
    static void vClearCoord(unsigned int size, void* res)
    {   MechanicalObjectCudaRigid3d_vClearCoord(size, res); }
    static void vClearDeriv(unsigned int size, void* res)
    {   MechanicalObjectCudaRigid3d_vClearDeriv(size, res); }
    static void vMEqCoord(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaRigid3d_vMEqCoord(size, res, f); }
    static void vMEqDeriv(unsigned int size, void* res, double f)
    {   MechanicalObjectCudaRigid3d_vMEqDeriv(size, res, f); }
    static void vEqBFCoord(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vEqBFCoord(size, res, b, f); }
    static void vEqBFDeriv(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vEqBFDeriv(size, res, b, f); }
    static void vPEqCoord(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3d_vPEqCoord(size, res, a); }
    static void vPEqCoordDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3d_vPEqCoordDeriv(size, res, a); }
    static void vPEqDeriv(unsigned int size, void* res, const void* a)
    {   MechanicalObjectCudaRigid3d_vPEqDeriv(size, res, a); }
    static void vPEqBFCoord(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vPEqBFCoord(size, res, b, f); }
    static void vPEqBFCoordDeriv(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vPEqBFCoordDeriv(size, res, b, f); }
    static void vPEqBFDeriv(unsigned int size, void* res, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vPEqBFDeriv(size, res, b, f); }
    static void vAddCoord(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3d_vAddCoord(size, res, a, b); }
    static void vAddCoordDeriv(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3d_vAddCoordDeriv(size, res, a, b); }
    static void vAddDeriv(unsigned int size, void* res, const void* a, const void* b)
    {   MechanicalObjectCudaRigid3d_vAddDeriv(size, res, a, b); }
    static void vOpCoord(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vOpCoord(size, res, a, b, f); }
    static void vOpCoordDeriv(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vOpCoordDeriv(size, res, a, b, f); }
    static void vOpDeriv(unsigned int size, void* res, const void* a, const void* b, double f)
    {   MechanicalObjectCudaRigid3d_vOpDeriv(size, res, a, b, f); }
    static void vIntegrate(unsigned int /* size */, const void* /* a */, void* /* v */, void* /* x */, double /* f_v_v */, double /* f_v_a */, double /* f_x_x */, double /* f_x_v */)
    { /*  MechanicalObjectCudaRigid3d_vIntegrate(size, a, v, x, f_v_v, f_v_a, f_x_x, f_x_v); */ }
    static void vPEqBF2(unsigned int /* size */, void* /* res1 */, const void* /* b1 */, double /* f1 */, void* /* res2 */, const void* /* b2 */, double /* f2 */)
    { /* MechanicalObjectCudaRigid3d_vPEqBF2(size, res1, b1, f1, res2, b2, f2); */ }
    static void vPEq4BF2(unsigned int /* size */, void* /* res1 */, const void* /* b11 */, double /* f11 */, const void* /* b12 */, double /* f12 */, const void* /* b13 */, double /* f13 */, const void* /* b14 */, double /* f14 */,
            void* /* res2 */, const void* /* b21 */, double /* f21 */, const void* /* b22 */, double /* f22 */, const void* /* b23 */, double /* f23 */, const void* /* b24 */, double /* f24 */)
    {
        /*   MechanicalObjectCudaRigid3d_vPEq4BF2(size, res1, b11, f11, b12, f12, b13, f13, b14, f14,
                                                    res2, b21, f21, b22, f22, b23, f23, b24, f24); */
    }
    static void vOp2(unsigned int /* size */, void* /* res1 */, const void* /* a1 */, const void* /* b1 */, double /* f1 */, void* /* res2 */, const void* /* a2 */, const void* /* b2 */, double /* f2 */)
    { /*  MechanicalObjectCudaRigid3d_vOp2(size, res1, a1, b1, f1, res2, a2, b2, f2); */ }
    static int vDotTmpSize(unsigned int size)
    {   return MechanicalObjectCudaRigid3d_vDotTmpSize(size); }
    static void vDot(unsigned int size, double* res, const void* a, const void* b, void* tmp, double* cputmp)
    {   MechanicalObjectCudaRigid3d_vDot(size, res, a, b, tmp, cputmp); }
    static void vMultiOpA1B3(unsigned int /*size*/, void* /*res1*/, const void* /*a1*/, void* /*res2*/, const void* /*a21*/, double /*f21*/, const void* /*a22*/, double /*f22*/, const void* /*a23*/, double /*f23*/)
    {  /* MechanicalObjectCudaRigid3d_vMultiOpA1B3(size, res1, a1, res2, a21, f21, a22, f22, a23, f23);*/ }
    static void vOp4(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, double /*f11*/, const void* /*a12*/, double /*f12*/, const void* /*a13*/, double /*f13*/, const void* /*a14*/, double /*f14*/)
    {  /* MechanicalObjectCudaRigid3d_vOp4(size, res1, a11, f11, a12, f12, a13, f13, a14, f14);*/ }
    static void vOpMCNAB(unsigned int /*size*/, void* /*res1*/, const void* /*a11*/, const void* /*a12*/, double /*f12*/, void* /*res2*/, const void* /*a21*/, void* /*res3*/, const void* /*a31*/)
    {  /* MechanicalObjectCudaRigid3d_vOpMCNAB(size, res1, a11, a12, f12, res2, a21, res3, a31);*/ }
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
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::accumulateForce(Main* m)
{
    if (!m->externalForces.getValue().empty())
    {
        //std::cout << "ADD: external forces, size = "<< m->externalForces->size() << std::endl;
        Kernels::vAssign(m->externalForces.getValue().size(),m->f.beginEdit()->deviceWrite(),m->externalForces.getValue().deviceRead());
        m->f.endEdit();
    }
    //else std::cout << "NO external forces" << std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vAlloc(Main* m, VecId v)
{
    if (v.type == sofa::core::V_COORD && v.index >= core::VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = m->getVecCoord(v.index);
        vec->recreate(m->vsize);
    }
    else if (v.type == sofa::core::V_DERIV && v.index >= core::VecDerivId::V_FIRST_DYNAMIC_INDEX)
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
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, double f)
{
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
                vv->recreate(m->vsize);
                Kernels::vClear(vv->size(), vv->deviceWrite());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
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
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops)
{
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

        void* w0Ptr, *r0Ptr0, *r0Ptr1;
        void* w1Ptr, *r1Ptr0, *r1Ptr1;

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

        void* w0Ptr, *r0Ptr[4];
        void* w1Ptr, *r1Ptr[4];

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
    // optimize a 3-way accumulation and a 1-way accumulation (BDF Explicit Solver) x1 = a1, x2 = a21*f21 + a22*f22 + a23*f23
    else if (ops.size()==2
            && ops[0].second.size()==1
            && ops[0].second[0].second == 1.0
            && ops[1].second.size()==3
            )
    {
        const unsigned int n = m->getSize();

        void* w0Ptr, *r0Ptr0;
        void* w1Ptr, *r1Ptr[3];

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        r0Ptr0 = (ops[0].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[0].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[0].first.getId(m)))->getValue().deviceRead();

        w1Ptr  = (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();
        r1Ptr[0] = (ops[1].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[0].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[0].first.getId(m)))->getValue().deviceRead();
        r1Ptr[1] = (ops[1].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[1].first.getId(m)))->getValue().deviceRead();
        r1Ptr[2] = (ops[1].second[2].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[1].second[2].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[1].second[2].first.getId(m)))->getValue().deviceRead();

        Kernels::vMultiOpA1B3(n, w0Ptr, r0Ptr0, w1Ptr, r1Ptr[0], ops[1].second[0].second, r1Ptr[1], ops[1].second[1].second, r1Ptr[2], ops[1].second[2].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[0].first.getId(m)))->endEdit();
        (ops[1].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[1].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[1].first.getId(m)))->endEdit();
    }
    // optimize a 4-way accumulation x = a11*f11 + a12*f12 + a13*f13 + a14*f14
    else if (ops.size()==1
            && ops[0].second.size()==4)
    {
        const unsigned int n = m->getSize();

        void* w0Ptr, *r0Ptr[4];

        w0Ptr  = (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite() : m->write(core::VecDerivId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        r0Ptr[0] = (ops[0].second[0].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[0].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[0].first.getId(m)))->getValue().deviceRead();
        r0Ptr[1] = (ops[0].second[1].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[1].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[1].first.getId(m)))->getValue().deviceRead();
        r0Ptr[2] = (ops[0].second[2].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[2].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[2].first.getId(m)))->getValue().deviceRead();
        r0Ptr[3] = (ops[0].second[3].first.getId(m).type == sofa::core::V_COORD) ? m->read(core::ConstVecCoordId(ops[0].second[3].first.getId(m)))->getValue().deviceRead() : m->read(core::ConstVecDerivId(ops[0].second[3].first.getId(m)))->getValue().deviceRead();

        Kernels::vOp4(n, w0Ptr, r0Ptr[0], ops[0].second[0].second, r0Ptr[1], ops[0].second[1].second, r0Ptr[2], ops[0].second[2].second, r0Ptr[3], ops[0].second[3].second);

        (ops[0].first.getId(m).type == sofa::core::V_COORD) ? m->write(core::VecCoordId(ops[0].first.getId(m)))->endEdit() : m->write(core::VecDerivId(ops[0].first.getId(m)))->endEdit();
    }
    // optimize the solve for MCNAB sovler
    // 1) x += v*dt
    // 2) prevReac=reac
    // 3) prevDiff=diff

    else if (ops.size()==3
            && ops[0].second.size()==2
            && ops[0].first.getId(m).type == sofa::core::V_COORD
            && ops[0].first.getId(m) == ops[0].second[0].first.getId(m)
            && ops[0].second[0].second == 1.0
            && ops[0].second[1].first.getId(m).type == sofa::core::V_DERIV
            && ops[1].second.size()==1
            && ops[2].second.size()==1)
    {
        const unsigned int n = m->getSize();

        void* w0Ptr, *r0Ptr[2];
        void* w1Ptr, *r1Ptr;
        void* w2Ptr, *r2Ptr;


        w0Ptr    = m->write(core::VecCoordId(ops[0].first.getId(m)))->beginEdit()->deviceWrite();
        r0Ptr[0] = m->read(core::ConstVecCoordId(ops[0].second[0].first.getId(m)))->getValue().deviceRead();
        r0Ptr[1] = m->read(core::ConstVecDerivId(ops[0].second[1].first.getId(m)))->getValue().deviceRead();

        w1Ptr    = m->write(core::VecDerivId(ops[1].first.getId(m)))->beginEdit()->deviceWrite();
        r1Ptr    = m->read(core::ConstVecDerivId(ops[1].second[0].first.getId(m)))->getValue().deviceRead();

        w2Ptr    = m->write(core::VecDerivId(ops[2].first.getId(m)))->beginEdit()->deviceWrite();
        r2Ptr    = m->read(core::ConstVecDerivId(ops[2].second[0].first.getId(m)))->getValue().deviceRead();

        Kernels::vOpMCNAB(n, w0Ptr, r0Ptr[0], r0Ptr[1], ops[0].second[1].second, w1Ptr, r1Ptr, w2Ptr, r2Ptr);

        m->write(core::VecCoordId(ops[0].first.getId(m)))->endEdit();
        m->write(core::VecDerivId(ops[1].first.getId(m)))->endEdit();
        m->write(core::VecDerivId(ops[2].first.getId(m)))->endEdit();
    }
    else // no optimization for now for other cases
    {
        std::cout << "size ops"<< ops.size() << "size second0"<< ops[0].second.size() << "size second1"<< ops[1].second.size() << std::endl;
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
            std::cout << std::endl;
        }
        {
            using namespace sofa::core::behavior;
            m->BaseMechanicalState::vMultiOp(params, ops);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
double MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::vDot(Main* m, ConstVecId a, ConstVecId b)
{
    Real r = 0.0f;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        const VecCoord* va = &m->read(core::ConstVecCoordId(a))->getValue();
        const VecCoord* vb = &m->read(core::ConstVecCoordId(b))->getValue();
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
        const VecDeriv* va = &m->read(core::ConstVecDerivId(a))->getValue();
        const VecDeriv* vb = &m->read(core::ConstVecDerivId(b))->getValue();
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
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::resetForce(Main* m)
{
    Data<VecDeriv>* d_f = m->write(core::VecDerivId::force());
    VecDeriv& f = *d_f->beginEdit();
    if (f.size() > 0)
        Kernels::vClear(f.size(), f.deviceWrite());
    d_f->endEdit();
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyToBaseVector(Main* m, defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        const VecCoord& vSrc = m->read(core::ConstVecCoordId(src))->getValue();

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
        const VecDeriv& vSrc = m->read(core::ConstVecDerivId(src))->getValue();

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
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyToCudaBaseVector(Main* m, sofa::gpu::cuda::CudaBaseVectorType<Real> * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        const VecCoord& va = m->read(core::ConstVecCoordId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;
        dest->invalidateHost();
        Kernels::vAssign(nbEntries, dest->deviceWrite(), va.deviceRead(offset*elemDim));

// 		offset += va->size() * elemDim;
    }
    else
    {
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        const VecCoord& va = m->read(core::ConstVecDerivId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;
        dest->invalidateHost();
        Kernels::vAssign(nbEntries, dest->deviceWrite(), ((Real *) va.deviceRead())+(offset*elemDim));

// 		offset += va->size() * elemDim;
    }
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyFromBaseVector(Main* m, VecId dest, const defaulttype::BaseVector * src, unsigned int &offset)
{

    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();

        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                DataTypeInfo<Coord>::setValue((*vDest)[i],j, src->element(offset + i * coordDim + j));
            }
        }

// 		offset += vDest->size() * coordDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
        VecDeriv* vDest = d_vDest->beginEdit();

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                DataTypeInfo<Deriv>::setValue((*vDest)[i], j, src->element(offset + i * derivDim + j));
            }
        }
// 		offset += vDest->size() * derivDim;
        d_vDest->endEdit();
    }
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::copyFromCudaBaseVector(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> * src,  unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/elemDim;
        vDest->invalidateHost();
        Kernels::vAssign(nbEntries, vDest->deviceWriteAt(offset*elemDim), src->deviceRead() );

// 		offset += vDest->size() * elemDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
        VecDeriv* vDest = d_vDest->beginEdit();
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/elemDim;
        vDest->invalidateHost();
        Kernels::vAssign(nbEntries, vDest->deviceWriteAt(offset*elemDim), src->deviceRead());

// 		offset += vDest->size() * elemDim;
        d_vDest->endEdit();
    }
}

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addFromBaseVectorSameSize(Main* m, VecId dest, const defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();

        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<coordDim; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Coord>::setValue((*vDest)[i], j, tmp + src->element(offset + i * coordDim + j));
            }
        }
// 		offset += vDest->size() * coordDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
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
// 		offset += vDest->size() * derivDim;
        d_vDest->endEdit();
    }
};

template<class TCoord, class TDeriv, class TReal>
void MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addFromCudaBaseVectorSameSize(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        Data<VecCoord>* d_va = m->write(core::VecCoordId(dest));
        VecCoord* va = d_va->beginEdit();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vPEq(nbEntries, va->deviceWrite(), src->deviceRead(offset*elemDim));

        offset += va->size() * elemDim;
        d_va->endEdit();
    }
    else
    {

        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        Data<VecDeriv>* d_va = m->write(core::VecDerivId(dest));
        VecDeriv* va = d_va->beginEdit();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vPEq(nbEntries, va->deviceWrite(), src->deviceRead(offset*elemDim));

        offset += va->size() * elemDim;
        d_va->endEdit();
    }
};

////////////////////////////////////
// Rigid Part
////////////////////////////////////
template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::accumulateForce(Main* m)
{
    if (!m->externalForces.getValue().empty())
    {
        //std::cout << "ADD: external forces, size = "<< m->externalForces->size() << std::endl;
        Kernels::vAssignDeriv(m->externalForces.getValue().size(),m->f.beginEdit()->deviceWrite(),m->externalForces.getValue().deviceRead());
        m->f.endEdit();
    }
    //else std::cout << "NO external forces" << std::endl;
}


template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::vAlloc(Main* m, VecId v)
{
    if (v.type == sofa::core::V_COORD && v.index >= core::VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = m->getVecCoord(v.index);
        vec->recreate(m->vsize);
    }
    else if (v.type == sofa::core::V_DERIV && v.index >= core::VecDerivId::V_FIRST_DYNAMIC_INDEX)
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

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::vOp(Main* m, VecId v, ConstVecId a, ConstVecId b, double f)
{
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
                vv->recreate(m->vsize);
                Kernels::vClearCoord(vv->size(), vv->deviceWrite());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                VecDeriv* vv = d_vv->beginEdit();
                vv->recreate(m->vsize);
                Kernels::vClearDeriv(vv->size(), vv->deviceWrite());
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
                    Kernels::vMEqCoord(vv->size(), vv->deviceWrite(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                    VecDeriv* vv = d_vv->beginEdit();
                    Kernels::vMEqDeriv(vv->size(), vv->deviceWrite(), (Real) f);
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
                    Kernels::vEqBFCoord(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
                    d_vv->endEdit();
                }
                else
                {
                    Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                    const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                    VecDeriv* vv = d_vv->beginEdit();
                    const VecDeriv* vb = &d_vb->getValue();
                    vv->recreate(vb->size());
                    Kernels::vEqBFDeriv(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real) f);
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
                Kernels::vAssignCoord(vv->size(), vv->deviceWrite(), va->deviceRead());
                d_vv->endEdit();
            }
            else
            {
                Data<VecDeriv>* d_vv = m->write((core::VecDerivId)v);
                const Data<VecDeriv>* d_va = m->read((core::ConstVecDerivId)a);
                VecDeriv* vv = d_vv->beginEdit();
                const VecDeriv* va = &d_va->getValue();
                vv->recreate(va->size());
                Kernels::vAssignDeriv(vv->size(), vv->deviceWrite(), va->deviceRead());
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
                            if (vb->size() > 0)
                                Kernels::vPEqCoord(vb->size(), vv->deviceWrite(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            if (vb->size() > vv->size())
                                vv->resize(vb->size());
                            if (vb->size() > 0)
                                Kernels::vPEqCoordDeriv(vb->size(), vv->deviceWrite(), vb->deviceRead());
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
                            Kernels::vPEqDeriv(vb->size(), vv->deviceWrite(), vb->deviceRead());
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
                            Kernels::vPEqBFCoord(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            vv->resize(vb->size());
                            Kernels::vPEqBFCoordDeriv(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
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
                        Kernels::vPEqBFDeriv(vv->size(), vv->deviceWrite(), vb->deviceRead(), (Real)f);
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
                            Kernels::vAddCoord(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vAddCoordDeriv(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
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
                        Kernels::vAddDeriv(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead());
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
                            Kernels::vOpCoord(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
                        }
                        else
                        {
                            const Data<VecDeriv>* d_vb = m->read((core::ConstVecDerivId)b);
                            const VecDeriv* vb = &d_vb->getValue();
                            Kernels::vOpCoordDeriv(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
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
                        Kernels::vOpDeriv(vv->size(), vv->deviceWrite(), va->deviceRead(), vb->deviceRead(), (Real)f);
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

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::vMultiOp(Main* m, const core::ExecParams* params, const VMultiOp& ops)
{
#ifdef DEBUG
	std::cerr<<"MechanicalObjectInternalData::vMultiOp currently not implemented for CudaRigidTypes !"<<std::endl;
#endif
    // TODO : make corresponding kernels

    // optimize common integration case: v += a*dt, x += v*dt
//     if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].first.type == sofa::core::V_DERIV && ops[0].second[1].first.type == sofa::core::V_DERIV
// 	                && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[0].first == ops[1].second[1].first && ops[1].first.type == sofa::core::V_COORD)
//     {
// 	VecDeriv* va = m->getVecDeriv(ops[0].second[1].first.index);
// 	VecDeriv* vv = m->getVecDeriv(ops[0].first.index);
// 	VecCoord* vx = m->getVecCoord(ops[1].first.index);
// 	const unsigned int n = vx->size();
// 	const double f_v_v = ops[0].second[0].second;
// 	const double f_v_a = ops[0].second[1].second;
// 	const double f_x_x = ops[1].second[0].second;
// 	const double f_x_v = ops[1].second[1].second;
// 	Kernels::vIntegrate(n, va->deviceRead(), vv->deviceWrite(), vx->deviceWrite(), (Real)f_v_v, (Real)f_v_a, (Real)f_x_x, (Real)f_x_v);
//     }
//     // optimize common CG step: x += a*p, q -= a*v
//     else if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].second[0].second == 1.0 && ops[0].first.type == sofa::core::V_DERIV && ops[0].second[1].first.type == sofa::core::V_DERIV
//                              && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[1].second[0].second == 1.0 && ops[1].first.type == sofa::core::V_DERIV && ops[1].second[1].first.type == sofa::core::V_DERIV)
//     {
//         VecDeriv* vv1 = m->getVecDeriv(ops[0].second[1].first.index);
//         VecDeriv* vres1 = m->getVecDeriv(ops[0].first.index);
//         VecDeriv* vv2 = m->getVecDeriv(ops[1].second[1].first.index);
//         VecDeriv* vres2 = m->getVecDeriv(ops[1].first.index);
//         const unsigned int n = vres1->size();
//         const double f1 = ops[0].second[1].second;
//         const double f2 = ops[1].second[1].second;
//         Kernels::vPEqBF2(n, vres1->deviceWrite(), vv1->deviceRead(), f1, vres2->deviceWrite(), vv2->deviceRead(), f2);
//     }
//     // optimize a pair of generic vOps
//     else if (ops.size()==2 && ops[0].second.size()==2 && ops[0].second[0].second == 1.0 && ops[1].second.size()==2 && ops[1].second[0].second == 1.0)
//     {
//         const unsigned int n = m->getSize();
//         Kernels::vOp2(n,
//             (ops[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].first.index)->deviceWrite() : m->getVecDeriv(ops[0].first.index)->deviceWrite(),
//             (ops[0].second[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[0].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[0].first.index)->deviceRead(),
//             (ops[0].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[1].first.index)->deviceRead(),
//             ops[0].second[1].second,
//             (ops[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].first.index)->deviceWrite() : m->getVecDeriv(ops[1].first.index)->deviceWrite(),
//             (ops[1].second[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[0].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[0].first.index)->deviceRead(),
//             (ops[1].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[1].first.index)->deviceRead(),
//             ops[1].second[1].second);
//     }
//     // optimize a pair of 4-way accumulations (such as at the end of RK4)
//     else if (ops.size()==2 && ops[0].second.size()==5 && ops[0].second[0].first == ops[0].first && ops[0].second[0].second == 1.0 &&
//                               ops[1].second.size()==5 && ops[1].second[0].first == ops[1].first && ops[1].second[0].second == 1.0)
//     {
//         const unsigned int n = m->getSize();
//         Kernels::vPEq4BF2(n,
//             (ops[0].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].first.index)->deviceWrite() : m->getVecDeriv(ops[0].first.index)->deviceWrite(),
//             (ops[0].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[1].first.index)->deviceRead(),
//             ops[0].second[1].second,
//             (ops[0].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[2].first.index)->deviceRead(),
//             ops[0].second[2].second,
//             (ops[0].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[3].first.index)->deviceRead(),
//             ops[0].second[3].second,
//             (ops[0].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[0].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[0].second[4].first.index)->deviceRead(),
//             ops[0].second[4].second,
//             (ops[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].first.index)->deviceWrite() : m->getVecDeriv(ops[1].first.index)->deviceWrite(),
//             (ops[1].second[1].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[1].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[1].first.index)->deviceRead(),
//             ops[1].second[1].second,
//             (ops[1].second[2].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[2].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[2].first.index)->deviceRead(),
//             ops[1].second[2].second,
//             (ops[1].second[3].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[3].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[3].first.index)->deviceRead(),
//             ops[1].second[3].second,
//             (ops[1].second[4].first.type == sofa::core::V_COORD) ? m->getVecCoord(ops[1].second[4].first.index)->deviceRead() : m->getVecDeriv(ops[1].second[4].first.index)->deviceRead(),
//             ops[1].second[4].second);
//     }
//     else // no optimization for now for other cases
//     {
//       std::cout << "CUDA: unoptimized vMultiOp:"<<std::endl;
//         for (unsigned int i=0;i<ops.size();++i)
//         {
//             std::cout << ops[i].first << " =";
//             if (ops[i].second.empty())
//                 std::cout << "0";
//             else
//                 for (unsigned int j=0;j<ops[i].second.size();++j)
//                 {
//                     if (j) std::cout << " + ";
//                     std::cout << ops[i].second[j].first << "*" << ops[i].second[j].second;
//                 }
//             std::cout << endl;
//         }
    {
        using namespace sofa::core::behavior;
        m->BaseMechanicalState::vMultiOp(params, ops);
    }
//     }
}

template<int N, class real>
double MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::vDot(Main* m, ConstVecId a, ConstVecId b)
{
    Real r = 0.0f;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        const VecCoord* va = &m->read(core::ConstVecCoordId(a))->getValue();
        const VecCoord* vb = &m->read(core::ConstVecCoordId(b))->getValue();
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
        const VecDeriv* va = &m->read(core::ConstVecDerivId(a))->getValue();
        const VecDeriv* vb = &m->read(core::ConstVecDerivId(b))->getValue();
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

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::resetForce(Main* m)
{
    Data<VecDeriv>* d_f = m->write(core::VecDerivId::force());
    VecDeriv& f = *d_f->beginEdit();

    if (f.size() == 0) return;
    Kernels::vClearDeriv(f.size(), f.deviceWrite());
    d_f->endEdit();
}

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::copyToBaseVector(Main* m, defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        const VecCoord& vSrc = m->read(core::ConstVecCoordId(src))->getValue();

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
        const VecDeriv& vSrc = m->read(core::ConstVecDerivId(src))->getValue();

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

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::copyToCudaBaseVector(Main* m, sofa::gpu::cuda::CudaBaseVectorType<Real> * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        const VecCoord& va = m->read(core::ConstVecCoordId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;

        Kernels::vAssignCoord(nbEntries, dest->deviceWrite(), va.deviceRead(offset*elemDim));

// 		offset += va->size() * elemDim;
    }
    else
    {
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        const VecDeriv& va = m->read(core::ConstVecDerivId(src))->getValue();
        const unsigned int nbEntries = dest->size()/elemDim;

        Kernels::vAssignDeriv(nbEntries, dest->deviceWrite(), va.deviceRead(offset*elemDim));

// 		offset += va->size() * elemDim;
    }
}

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::copyFromBaseVector(Main* m, VecId dest, const defaulttype::BaseVector * src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();

        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<3; j++)
            {
                DataTypeInfo<Coord>::setValue((*vDest)[i],j,src->element(offset + i * coordDim + j));
            }
        }

// 	offset += vDest->size() * coordDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
        VecDeriv* vDest = d_vDest->beginEdit();

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<derivDim; j++)
            {
                DataTypeInfo<Deriv>::setValue((*vDest)[i], j, src->element(offset + i * derivDim + j));
            }
        }
// 	offset += vDest->size() * derivDim;
        d_vDest->endEdit();
    }
}

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::copyFromCudaBaseVector(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> * src,  unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
        VecCoord* vDest = d_vDest->beginEdit();
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vAssignCoord(nbEntries, vDest->deviceWriteAt(offset*elemDim), src->deviceRead() );

// 		offset += vDest->size() * elemDim;
        d_vDest->endEdit();
    }
    else
    {
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
        VecDeriv* vDest = d_vDest->beginEdit();
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vAssignDeriv(nbEntries, vDest->deviceWriteAt(offset*elemDim), src->deviceRead());

// 		offset += vDest->size() * elemDim;
        d_vDest->endEdit();
    }
}

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addFromBaseVectorSameSize(Main* m, VecId dest, const defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        Data<VecCoord>* d_vDest = m->write(core::VecCoordId(dest));
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
        Data<VecDeriv>* d_vDest = m->write(core::VecDerivId(dest));
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
}

template<int N, class real>
void MechanicalObjectInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addFromCudaBaseVectorSameSize(Main* m, VecId dest, const sofa::gpu::cuda::CudaBaseVectorType<Real> *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        unsigned int elemDim = DataTypeInfo<Coord>::size();
        Data<VecCoord>* d_va = m->write(core::VecCoordId(dest));
        VecCoord* va = d_va->beginEdit();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vPEqDeriv(nbEntries, va->deviceWrite(), src->deviceRead(offset*elemDim));

        offset += va->size() * elemDim;
        d_va->endEdit();
    }
    else
    {
        unsigned int elemDim = DataTypeInfo<Deriv>::size();
        Data<VecDeriv>* d_va = m->write(core::VecDerivId(dest));
        VecDeriv* va = d_va->beginEdit();
        const unsigned int nbEntries = src->size()/elemDim;

        Kernels::vPEqDeriv(nbEntries, va->deviceWrite(), src->deviceRead(offset*elemDim));

        offset += va->size() * elemDim;
        d_va->endEdit();
    }
}



// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_ImplMethods(T) \
template<> void MechanicalObject< T >::accumulateForce(const core::ExecParams* params, core::VecDerivId fid) \
{ if( fid==core::VecDerivId::force() ) data.accumulateForce(this); else core::behavior::BaseMechanicalState::accumulateForce(params,fid);} \
template<> void MechanicalObject< T >::vOp(const core::ExecParams* /* params */, core::VecId v, core::ConstVecId a, core::ConstVecId b, SReal f) \
{ data.vOp(this, v, a, b, f); }		\
template<> void MechanicalObject< T >::vMultiOp(const core::ExecParams* params, const VMultiOp& ops) \
{ data.vMultiOp(this, params, ops); }                                    \
template<> SReal MechanicalObject< T >::vDot(const core::ExecParams* /* params */, core::ConstVecId a, core::ConstVecId b) \
{ return data.vDot(this, a, b); }				    \
template<> void MechanicalObject< T >::resetForce(const core::ExecParams* params, core::VecDerivId fid) \
{ if( fid==core::VecDerivId::force() ) data.resetForce(this); else core::behavior::BaseMechanicalState::resetForce(params,fid); } \
template<> void MechanicalObject< T >::copyToBaseVector(defaulttype::BaseVector * dest, core::ConstVecId src, unsigned int &offset) \
{ if (CudaBaseVectorType<Real> * vec = dynamic_cast<CudaBaseVectorType<Real> *>(dest)) data.copyToCudaBaseVector(this, vec,src,offset); \
else data.copyToBaseVector(this, dest,src,offset); } \
template<> void MechanicalObject< T >::copyFromBaseVector(core::VecId dest, const defaulttype::BaseVector * src, unsigned int &offset) \
{ if (const CudaBaseVectorType<Real> * vec = dynamic_cast<const CudaBaseVectorType<Real> *>(src)) data.copyFromCudaBaseVector(this, dest,vec,offset); \
else data.copyFromBaseVector(this, dest,src,offset); } \
template<> void MechanicalObject< T >::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector *src, unsigned int &offset) \
{ if (const CudaBaseVectorType<Real> * vec = dynamic_cast<const CudaBaseVectorType<Real> *>(src)) data.addFromCudaBaseVectorSameSize(this, dest,vec,offset); \
else data.addFromBaseVectorSameSize(this, dest,src,offset); }

CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec1fTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec2fTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3fTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3f1Types)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec6fTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaRigid3fTypes)


#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3dTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec3d1Types)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaVec6dTypes)
CudaMechanicalObject_ImplMethods(gpu::cuda::CudaRigid3dTypes)

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaMechanicalObject_ImplMethods








}

} // namespace component

} // namespace sofa

#endif
