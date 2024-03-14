/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>

#include <cassert>

#include "cuda.h"
#include "mycuda.h"

using namespace sofa::gpu::cuda;

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
    void CudaTetrahedronTLEDForceField3f_addForce(int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* volume, int2* forceCoordinates, float3*
                                                  preferredDirection, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2, float4* F0, float4* F1, float4* F2, float4* F3, float
                                                  Lambda, float Mu, unsigned nbElem, unsigned nbVertex, unsigned nbElemPerVertex, unsigned viscoelasticity, unsigned
                                                  anisotropy, const float* x, const float* x0, void* f);
    void InitGPU_TetrahedronTLED(int valence, int nbVertex, int nbElements);
    void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv);
    void InitGPU_TetrahedronAniso();
    void ClearGPU_TetrahedronTLED(void);
    void ClearGPU_TetrahedronVisco(void);
    void ClearGPU_TetrahedronAniso(void);
}

/**
 * GPU-side methods
 */

/**
 * By default, we use viscoelastic isochoric terms only for simplicity
 * We define a hard-coded directive to save registers in the kernel (instead of a boolean to switch to the desired term in the kernel)
 */
#define ISOCHORIC

/**
 * Constant memory
 */
// A few constants for viscoelasticity
static __constant__ float Ai_gpu[2];
static __constant__ float Av_gpu[2];

// A material constant used for the transverse isotropy
static __constant__ int Eta_gpu;

// Function to be called from the device to compute forces from stresses (Prototype)
__device__ float4 computeForce_tet(const int node, const float4 DhC0, const float4 DhC1, const float4 DhC2,
        const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp,
        const float3 Node4Disp, const float * SPK, const int tid);

// Device function returning the position of vertex i as a float3
__device__ CudaVec3f getX(const float* x, int i)
{
    const int i3 = i * 3;
    const float x1 = x[i3];
    const float x2 = x[i3+1];
    const float x3 = x[i3+2];
    return CudaVec3f::make(x1,x2,x3);
}

/**
 * This version is valid for tetrahedral meshes and uses an elastic formulation
 */
__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet0(
    const float* x, const float* x0,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* volume, float Lambda,
    float Mu, int nbElem, float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
        float4 Dh0 = DhC0[index];
        float4 Dh1 = DhC1[index];
        float4 Dh2 = DhC2[index];

        int4 NodesPerElement = nodesPerElement[index];
        CudaVec3f Node1Disp = getX(x, NodesPerElement.x) - getX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = getX(x, NodesPerElement.y) - getX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = getX(x, NodesPerElement.z) - getX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = getX(x, NodesPerElement.w) - getX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] = Dh0.x*Node1Disp.x + Dh0.y*Node2Disp.x + Dh0.z*Node3Disp.x + Dh0.w*Node4Disp.x + 1.0f;
        XT[1][0] = Dh1.x*Node1Disp.x + Dh1.y*Node2Disp.x + Dh1.z*Node3Disp.x + Dh1.w*Node4Disp.x;
        XT[2][0] = Dh2.x*Node1Disp.x + Dh2.y*Node2Disp.x + Dh2.z*Node3Disp.x + Dh2.w*Node4Disp.x;

        ///Column 2
        XT[0][1] = Dh0.x*Node1Disp.y + Dh0.y*Node2Disp.y + Dh0.z*Node3Disp.y + Dh0.w*Node4Disp.y;
        XT[1][1] = Dh1.x*Node1Disp.y + Dh1.y*Node2Disp.y + Dh1.z*Node3Disp.y + Dh1.w*Node4Disp.y + 1.0f;
        XT[2][1] = Dh2.x*Node1Disp.y + Dh2.y*Node2Disp.y + Dh2.z*Node3Disp.y + Dh2.w*Node4Disp.y;

        ///Column 3
        XT[0][2] = Dh0.x*Node1Disp.z + Dh0.y*Node2Disp.z + Dh0.z*Node3Disp.z + Dh0.w*Node4Disp.z;
        XT[1][2] = Dh1.x*Node1Disp.z + Dh1.y*Node2Disp.z + Dh1.z*Node3Disp.z + Dh1.w*Node4Disp.z;
        XT[2][2] = Dh2.x*Node1Disp.z + Dh2.y*Node2Disp.z + Dh2.z*Node3Disp.z + Dh2.w*Node4Disp.z + 1.0f;

        /**
        * Computes the right Cauchy-Green deformation tensor C = XT*X (in fact we compute only 6 terms since C is symetric)
        */
        float C11, C12, C13, C22, C23, C33;
        C11 = XT[0][0]*XT[0][0] + XT[0][1]*XT[0][1] + XT[0][2]*XT[0][2];
        C12 = XT[0][0]*XT[1][0] + XT[0][1]*XT[1][1] + XT[0][2]*XT[1][2];
        C13 = XT[0][0]*XT[2][0] + XT[0][1]*XT[2][1] + XT[0][2]*XT[2][2];
        C22 = XT[1][0]*XT[1][0] + XT[1][1]*XT[1][1] + XT[1][2]*XT[1][2];
        C23 = XT[1][0]*XT[2][0] + XT[1][1]*XT[2][1] + XT[1][2]*XT[2][2];
        C33 = XT[2][0]*XT[2][0] + XT[2][1]*XT[2][1] + XT[2][2]*XT[2][2];


        /**
        * Computes determinant of X
        */
        float J = XT[0][0]*( XT[1][1]*XT[2][2] - XT[2][1]*XT[1][2] )
                - XT[1][0]*( XT[0][1]*XT[2][2] - XT[2][1]*XT[0][2] )
                + XT[2][0]*( XT[0][1]*XT[1][2] - XT[1][1]*XT[0][2] );

        /**
        * Computes second Piola-Kirchoff stress
        */
        float SPK[6];

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverses
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        /// Isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        float x4 = __fdividef(-x1*(C11+C22+C33), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x4*Ci11 + x1;
        SiE22 = x4*Ci22 + x1;
        SiE33 = x4*Ci33 + x1;
        SiE12 = x4*Ci12;
        SiE23 = x4*Ci23;
        SiE13 = x4*Ci13;

        float SvE11, SvE12, SvE13, SvE22, SvE23, SvE33;
        SvE11 = x5*Ci11;
        SvE22 = x5*Ci22;
        SvE33 = x5*Ci33;
        SvE12 = x5*Ci12;
        SvE23 = x5*Ci23;
        SvE13 = x5*Ci13;

        SPK[0] = SiE11 + SvE11;
        SPK[1] = SiE22 + SvE22;
        SPK[2] = SiE33 + SvE33;
        SPK[3] = SiE12 + SvE12;
        SPK[4] = SiE23 + SvE23;
        SPK[5] = SiE13 + SvE13;

        /// Retrieves the volume
        float Vol = volume[index];
        SPK[0] *= Vol;
        SPK[1] *= Vol;
        SPK[2] *= Vol;
        SPK[3] *= Vol;
        SPK[4] *= Vol;
        SPK[5] *= Vol;


        /**
         * Computes strain-displacement matrix and writes the result in global memory
         */
        F0[index] = computeForce_tet(0, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F1[index] = computeForce_tet(1, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F2[index] = computeForce_tet(2, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F3[index] = computeForce_tet(3, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);

    }

}

/**
 * This version is valid for tetrahedral meshes and uses a transversely isotropic and elastic formulation
 */
__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet1(
    const float* x, const float* x0,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* volume,
    float3* preferredDirection, float Lambda, float Mu, int nbElem, float4* F0, float4* F1,
    float4* F2, float4* F3)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
        float4 Dh0 = DhC0[index];
        float4 Dh1 = DhC1[index];
        float4 Dh2 = DhC2[index];

        int4 NodesPerElement = nodesPerElement[index];
        CudaVec3f Node1Disp = getX(x, NodesPerElement.x) - getX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = getX(x, NodesPerElement.y) - getX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = getX(x, NodesPerElement.z) - getX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = getX(x, NodesPerElement.w) - getX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] = Dh0.x*Node1Disp.x + Dh0.y*Node2Disp.x + Dh0.z*Node3Disp.x + Dh0.w*Node4Disp.x + 1.0f;
        XT[1][0] = Dh1.x*Node1Disp.x + Dh1.y*Node2Disp.x + Dh1.z*Node3Disp.x + Dh1.w*Node4Disp.x;
        XT[2][0] = Dh2.x*Node1Disp.x + Dh2.y*Node2Disp.x + Dh2.z*Node3Disp.x + Dh2.w*Node4Disp.x;

        ///Column 2
        XT[0][1] = Dh0.x*Node1Disp.y + Dh0.y*Node2Disp.y + Dh0.z*Node3Disp.y + Dh0.w*Node4Disp.y;
        XT[1][1] = Dh1.x*Node1Disp.y + Dh1.y*Node2Disp.y + Dh1.z*Node3Disp.y + Dh1.w*Node4Disp.y + 1.0f;
        XT[2][1] = Dh2.x*Node1Disp.y + Dh2.y*Node2Disp.y + Dh2.z*Node3Disp.y + Dh2.w*Node4Disp.y;

        ///Column 3
        XT[0][2] = Dh0.x*Node1Disp.z + Dh0.y*Node2Disp.z + Dh0.z*Node3Disp.z + Dh0.w*Node4Disp.z;
        XT[1][2] = Dh1.x*Node1Disp.z + Dh1.y*Node2Disp.z + Dh1.z*Node3Disp.z + Dh1.w*Node4Disp.z;
        XT[2][2] = Dh2.x*Node1Disp.z + Dh2.y*Node2Disp.z + Dh2.z*Node3Disp.z + Dh2.w*Node4Disp.z + 1.0f;

        /**
        * Computes the right Cauchy-Green deformation tensor C = XT*X (in fact we compute only 6 terms since C is symetric)
        */
        float C11, C12, C13, C22, C23, C33;
        C11 = XT[0][0]*XT[0][0] + XT[0][1]*XT[0][1] + XT[0][2]*XT[0][2];
        C12 = XT[0][0]*XT[1][0] + XT[0][1]*XT[1][1] + XT[0][2]*XT[1][2];
        C13 = XT[0][0]*XT[2][0] + XT[0][1]*XT[2][1] + XT[0][2]*XT[2][2];
        C22 = XT[1][0]*XT[1][0] + XT[1][1]*XT[1][1] + XT[1][2]*XT[1][2];
        C23 = XT[1][0]*XT[2][0] + XT[1][1]*XT[2][1] + XT[1][2]*XT[2][2];
        C33 = XT[2][0]*XT[2][0] + XT[2][1]*XT[2][1] + XT[2][2]*XT[2][2];


        /**
        * Computes determinant of X
        */
        float J = XT[0][0]*( XT[1][1]*XT[2][2] - XT[2][1]*XT[1][2] )
                - XT[1][0]*( XT[0][1]*XT[2][2] - XT[2][1]*XT[0][2] )
                + XT[2][0]*( XT[0][1]*XT[1][2] - XT[1][1]*XT[0][2] );

        /**
        * Computes second Piola-Kirchoff stress
        */
        float SPK[6];

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverses
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        /// Transversely isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        // Bracketed term is I4 = A:C

        // Reads the preferred direction
        float3 a = preferredDirection[index];

        float x2 = J23*(a.x*a.x*C11 + a.y*a.y*C22 + a.z*a.z*C33 + 2*a.x*a.y*C12 + 2*a.y*a.z*C23 + 2*a.x*a.z*C13) - 1;
        float x3 = J23*Eta_gpu*x2;
        float x4 = __fdividef(-(Eta_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*a.x*a.x + x4*Ci11 + x1;
        SiE22 = x3*a.y*a.y + x4*Ci22 + x1;
        SiE33 = x3*a.z*a.z + x4*Ci33 + x1;
        SiE12 = x3*a.x*a.y + x4*Ci12;
        SiE23 = x3*a.y*a.z + x4*Ci23;
        SiE13 = x3*a.x*a.z + x4*Ci13;

        float SvE11, SvE12, SvE13, SvE22, SvE23, SvE33;
        SvE11 = x5*Ci11;
        SvE22 = x5*Ci22;
        SvE33 = x5*Ci33;
        SvE12 = x5*Ci12;
        SvE23 = x5*Ci23;
        SvE13 = x5*Ci13;

        SPK[0] = SiE11 + SvE11;
        SPK[1] = SiE22 + SvE22;
        SPK[2] = SiE33 + SvE33;
        SPK[3] = SiE12 + SvE12;
        SPK[4] = SiE23 + SvE23;
        SPK[5] = SiE13 + SvE13;

        /// Retrieves the volume
        float Vol = volume[index];
        SPK[0] *= Vol;
        SPK[1] *= Vol;
        SPK[2] *= Vol;
        SPK[3] *= Vol;
        SPK[4] *= Vol;
        SPK[5] *= Vol;


        /**
         * Computes strain-displacement matrix and writes the result in global memory
         */
        F0[index] = computeForce_tet(0, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F1[index] = computeForce_tet(1, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F2[index] = computeForce_tet(2, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F3[index] = computeForce_tet(3, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);


    }

}

/**
 * This version is valid for tetrahedral meshes and uses a viscoelastic formulation based on separated isochoric and volumetric terms. The model is isotropic.
 */
__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet2(
    const float* x, const float* x0,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* volume, float Lambda,
    float Mu, int nbElem, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2, float4* F0,
    float4* F1, float4* F2, float4* F3)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {

        /// Shape function derivatives matrix
        float4 Dh0 = DhC0[index];
        float4 Dh1 = DhC1[index];
        float4 Dh2 = DhC2[index];

        int4 NodesPerElement = nodesPerElement[index];
        CudaVec3f Node1Disp = getX(x, NodesPerElement.x) - getX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = getX(x, NodesPerElement.y) - getX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = getX(x, NodesPerElement.z) - getX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = getX(x, NodesPerElement.w) - getX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] = Dh0.x*Node1Disp.x + Dh0.y*Node2Disp.x + Dh0.z*Node3Disp.x + Dh0.w*Node4Disp.x + 1.0f;
        XT[1][0] = Dh1.x*Node1Disp.x + Dh1.y*Node2Disp.x + Dh1.z*Node3Disp.x + Dh1.w*Node4Disp.x;
        XT[2][0] = Dh2.x*Node1Disp.x + Dh2.y*Node2Disp.x + Dh2.z*Node3Disp.x + Dh2.w*Node4Disp.x;

        ///Column 2
        XT[0][1] = Dh0.x*Node1Disp.y + Dh0.y*Node2Disp.y + Dh0.z*Node3Disp.y + Dh0.w*Node4Disp.y;
        XT[1][1] = Dh1.x*Node1Disp.y + Dh1.y*Node2Disp.y + Dh1.z*Node3Disp.y + Dh1.w*Node4Disp.y + 1.0f;
        XT[2][1] = Dh2.x*Node1Disp.y + Dh2.y*Node2Disp.y + Dh2.z*Node3Disp.y + Dh2.w*Node4Disp.y;

        ///Column 3
        XT[0][2] = Dh0.x*Node1Disp.z + Dh0.y*Node2Disp.z + Dh0.z*Node3Disp.z + Dh0.w*Node4Disp.z;
        XT[1][2] = Dh1.x*Node1Disp.z + Dh1.y*Node2Disp.z + Dh1.z*Node3Disp.z + Dh1.w*Node4Disp.z;
        XT[2][2] = Dh2.x*Node1Disp.z + Dh2.y*Node2Disp.z + Dh2.z*Node3Disp.z + Dh2.w*Node4Disp.z + 1.0f;

        /**
        * Computes the right Cauchy-Green deformation tensor C = XT*X (in fact we compute only 6 terms since C is symetric)
        */
        float C11, C12, C13, C22, C23, C33;
        C11 = XT[0][0]*XT[0][0] + XT[0][1]*XT[0][1] + XT[0][2]*XT[0][2];
        C12 = XT[0][0]*XT[1][0] + XT[0][1]*XT[1][1] + XT[0][2]*XT[1][2];
        C13 = XT[0][0]*XT[2][0] + XT[0][1]*XT[2][1] + XT[0][2]*XT[2][2];
        C22 = XT[1][0]*XT[1][0] + XT[1][1]*XT[1][1] + XT[1][2]*XT[1][2];
        C23 = XT[1][0]*XT[2][0] + XT[1][1]*XT[2][1] + XT[1][2]*XT[2][2];
        C33 = XT[2][0]*XT[2][0] + XT[2][1]*XT[2][1] + XT[2][2]*XT[2][2];


        /**
        * Computes determinant of X
        */
        float J = XT[0][0]*( XT[1][1]*XT[2][2] - XT[2][1]*XT[1][2] )
                - XT[1][0]*( XT[0][1]*XT[2][2] - XT[2][1]*XT[0][2] )
                + XT[2][0]*( XT[0][1]*XT[1][2] - XT[1][1]*XT[0][2] );

        /**
        * Computes second Piola-Kirchoff stress
        */
        float SPK[6];

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverses
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        /// Isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        float x4 = __fdividef(-x1*(C11+C22+C33), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x4*Ci11 + x1;
        SiE22 = x4*Ci22 + x1;
        SiE33 = x4*Ci33 + x1;
        SiE12 = x4*Ci12;
        SiE23 = x4*Ci23;
        SiE13 = x4*Ci13;

        float SvE11, SvE12, SvE13, SvE22, SvE23, SvE33;
        SvE11 = x5*Ci11;
        SvE22 = x5*Ci22;
        SvE33 = x5*Ci33;
        SvE12 = x5*Ci12;
        SvE23 = x5*Ci23;
        SvE13 = x5*Ci13;

        SPK[0] = SiE11 + SvE11;
        SPK[1] = SiE22 + SvE22;
        SPK[2] = SiE33 + SvE33;
        SPK[3] = SiE12 + SvE12;
        SPK[4] = SiE23 + SvE23;
        SPK[5] = SiE13 + SvE13;

        /// Viscoelastic components of response
        float4 temp;

#ifdef ISOCHORIC
        // Isochoric part
        temp = Di1[index];
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_gpu[1]; temp.z += Ai_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_gpu[+1]; temp.w += Ai_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Di2[index];
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = Di1[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Di2[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Retrieves the volume
        float Vol = volume[index];
        SPK[0] *= Vol;
        SPK[1] *= Vol;
        SPK[2] *= Vol;
        SPK[3] *= Vol;
        SPK[4] *= Vol;
        SPK[5] *= Vol;


        /**
         * Computes strain-displacement matrix and writes the result in global memory
         */
        F0[index] = computeForce_tet(0, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F1[index] = computeForce_tet(1, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F2[index] = computeForce_tet(2, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F3[index] = computeForce_tet(3, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);


    }

}

/**
 * This version is valid for tetrahedral meshes and uses an viscoelastic and anisotropic formulation
 */
__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet3(
    const float* x, const float* x0,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* volume,
    float3* preferredDirection, float Lambda, float Mu, int nbElem, float4* Di1, float4* Di2,
    float4* Dv1, float4* Dv2, float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {

        /// Shape function derivatives matrix
        float4 Dh0 = DhC0[index];
        float4 Dh1 = DhC1[index];
        float4 Dh2 = DhC2[index];

        int4 NodesPerElement = nodesPerElement[index];
        CudaVec3f Node1Disp = getX(x, NodesPerElement.x) - getX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = getX(x, NodesPerElement.y) - getX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = getX(x, NodesPerElement.z) - getX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = getX(x, NodesPerElement.w) - getX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] = Dh0.x*Node1Disp.x + Dh0.y*Node2Disp.x + Dh0.z*Node3Disp.x + Dh0.w*Node4Disp.x + 1.0f;
        XT[1][0] = Dh1.x*Node1Disp.x + Dh1.y*Node2Disp.x + Dh1.z*Node3Disp.x + Dh1.w*Node4Disp.x;
        XT[2][0] = Dh2.x*Node1Disp.x + Dh2.y*Node2Disp.x + Dh2.z*Node3Disp.x + Dh2.w*Node4Disp.x;

        ///Column 2
        XT[0][1] = Dh0.x*Node1Disp.y + Dh0.y*Node2Disp.y + Dh0.z*Node3Disp.y + Dh0.w*Node4Disp.y;
        XT[1][1] = Dh1.x*Node1Disp.y + Dh1.y*Node2Disp.y + Dh1.z*Node3Disp.y + Dh1.w*Node4Disp.y + 1.0f;
        XT[2][1] = Dh2.x*Node1Disp.y + Dh2.y*Node2Disp.y + Dh2.z*Node3Disp.y + Dh2.w*Node4Disp.y;

        ///Column 3
        XT[0][2] = Dh0.x*Node1Disp.z + Dh0.y*Node2Disp.z + Dh0.z*Node3Disp.z + Dh0.w*Node4Disp.z;
        XT[1][2] = Dh1.x*Node1Disp.z + Dh1.y*Node2Disp.z + Dh1.z*Node3Disp.z + Dh1.w*Node4Disp.z;
        XT[2][2] = Dh2.x*Node1Disp.z + Dh2.y*Node2Disp.z + Dh2.z*Node3Disp.z + Dh2.w*Node4Disp.z + 1.0f;

        /**
        * Computes the right Cauchy-Green deformation tensor C = XT*X (in fact we compute only 6 terms since C is symetric)
        */
        float C11, C12, C13, C22, C23, C33;
        C11 = XT[0][0]*XT[0][0] + XT[0][1]*XT[0][1] + XT[0][2]*XT[0][2];
        C12 = XT[0][0]*XT[1][0] + XT[0][1]*XT[1][1] + XT[0][2]*XT[1][2];
        C13 = XT[0][0]*XT[2][0] + XT[0][1]*XT[2][1] + XT[0][2]*XT[2][2];
        C22 = XT[1][0]*XT[1][0] + XT[1][1]*XT[1][1] + XT[1][2]*XT[1][2];
        C23 = XT[1][0]*XT[2][0] + XT[1][1]*XT[2][1] + XT[1][2]*XT[2][2];
        C33 = XT[2][0]*XT[2][0] + XT[2][1]*XT[2][1] + XT[2][2]*XT[2][2];


        /**
        * Computes determinant of X
        */
        float J = XT[0][0]*( XT[1][1]*XT[2][2] - XT[2][1]*XT[1][2] )
                - XT[1][0]*( XT[0][1]*XT[2][2] - XT[2][1]*XT[0][2] )
                + XT[2][0]*( XT[0][1]*XT[1][2] - XT[1][1]*XT[0][2] );

        /**
        * Computes second Piola-Kirchoff stress
        */
        float SPK[6];

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverses
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        /// Transversely isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        // Bracketed term is I4 = A:C

        // Reads the preferred direction
        float3 a = preferredDirection[index];

        float x2 = J23*(a.x*a.x*C11 + a.y*a.y*C22 + a.z*a.z*C33 + 2*a.x*a.y*C12 + 2*a.y*a.z*C23 + 2*a.x*a.z*C13) - 1;
        float x3 = J23*Eta_gpu*x2;
        float x4 = __fdividef(-(Eta_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*a.x*a.x + x4*Ci11 + x1;
        SiE22 = x3*a.y*a.y + x4*Ci22 + x1;
        SiE33 = x3*a.z*a.z + x4*Ci33 + x1;
        SiE12 = x3*a.x*a.y + x4*Ci12;
        SiE23 = x3*a.y*a.z + x4*Ci23;
        SiE13 = x3*a.x*a.z + x4*Ci13;

        float SvE11, SvE12, SvE13, SvE22, SvE23, SvE33;
        SvE11 = x5*Ci11;
        SvE22 = x5*Ci22;
        SvE33 = x5*Ci33;
        SvE12 = x5*Ci12;
        SvE23 = x5*Ci23;
        SvE13 = x5*Ci13;

        SPK[0] = SiE11 + SvE11;
        SPK[1] = SiE22 + SvE22;
        SPK[2] = SiE33 + SvE33;
        SPK[3] = SiE12 + SvE12;
        SPK[4] = SiE23 + SvE23;
        SPK[5] = SiE13 + SvE13;

        /// Viscoelastic components of response
        float4 temp;

#ifdef ISOCHORIC
        // Isochoric part
        temp = Di1[index];
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_gpu[1]; temp.z += Ai_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_gpu[+1]; temp.w += Ai_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Di2[index];
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = Di1[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Di2[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Retrieves the volume
        float Vol = volume[index];
        SPK[0] *= Vol;
        SPK[1] *= Vol;
        SPK[2] *= Vol;
        SPK[3] *= Vol;
        SPK[4] *= Vol;
        SPK[5] *= Vol;


        /**
         * Computes strain-displacement matrix and writes the result in global memory
         */
        F0[index] = computeForce_tet(0, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F1[index] = computeForce_tet(1, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F2[index] = computeForce_tet(2, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);
        F3[index] = computeForce_tet(3, Dh0, Dh1, Dh2, Node1Disp, Node2Disp, Node3Disp, Node4Disp, SPK, index);


    }

}


/**
 * Function to be called from the device to compute forces from stresses
 * => allows us to save registers by recomputing the deformation gradient
 */
__device__ float4 computeForce_tet(const int node, const float4 DhC0, const float4 DhC1, const float4 DhC2,
        const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp,
        const float3 Node4Disp, const float * SPK, const int tid)
{
    float XT[3][3];

    ///Column 1
    XT[0][0] = DhC0.x*Node1Disp.x + DhC0.y*Node2Disp.x + DhC0.z*Node3Disp.x + DhC0.w*Node4Disp.x + 1.0f;
    XT[1][0] = DhC1.x*Node1Disp.x + DhC1.y*Node2Disp.x + DhC1.z*Node3Disp.x + DhC1.w*Node4Disp.x;
    XT[2][0] = DhC2.x*Node1Disp.x + DhC2.y*Node2Disp.x + DhC2.z*Node3Disp.x + DhC2.w*Node4Disp.x;

    ///Column 2
    XT[0][1] = DhC0.x*Node1Disp.y + DhC0.y*Node2Disp.y + DhC0.z*Node3Disp.y + DhC0.w*Node4Disp.y;
    XT[1][1] = DhC1.x*Node1Disp.y + DhC1.y*Node2Disp.y + DhC1.z*Node3Disp.y + DhC1.w*Node4Disp.y + 1.0f;
    XT[2][1] = DhC2.x*Node1Disp.y + DhC2.y*Node2Disp.y + DhC2.z*Node3Disp.y + DhC2.w*Node4Disp.y;

    ///Column 3
    XT[0][2] = DhC0.x*Node1Disp.z + DhC0.y*Node2Disp.z + DhC0.z*Node3Disp.z + DhC0.w*Node4Disp.z;
    XT[1][2] = DhC1.x*Node1Disp.z + DhC1.y*Node2Disp.z + DhC1.z*Node3Disp.z + DhC1.w*Node4Disp.z;
    XT[2][2] = DhC2.x*Node1Disp.z + DhC2.y*Node2Disp.z + DhC2.z*Node3Disp.z + DhC2.w*Node4Disp.z + 1.0f;


    float BL[6];
    float FX, FY, FZ;

    float Dh0, Dh1, Dh2;
    switch(node)
    {
    case 0 :
        Dh0 = DhC0.x;
        Dh1 = DhC1.x;
        Dh2 = DhC2.x;
        break;

    case 1 :
        Dh0 = DhC0.y;
        Dh1 = DhC1.y;
        Dh2 = DhC2.y;
        break;

    case 2 :
        Dh0 = DhC0.z;
        Dh1 = DhC1.z;
        Dh2 = DhC2.z;
        break;

    case 3 :
        Dh0 = DhC0.w;
        Dh1 = DhC1.w;
        Dh2 = DhC2.w;
        break;
    }


    /// Computes X component
    BL[0] = Dh0 * XT[0][0];
    BL[1] = Dh1 * XT[1][0];
    BL[2] = Dh2 * XT[2][0];
    BL[3] = Dh1 * XT[0][0] + Dh0 * XT[1][0];
    BL[4] = Dh2 * XT[1][0] + Dh1 * XT[2][0];
    BL[5] = Dh2 * XT[0][0] + Dh0 * XT[2][0];
    FX = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /// Computes Y component
    BL[0] = Dh0 * XT[0][1];
    BL[1] = Dh1 * XT[1][1];
    BL[2] = Dh2 * XT[2][1];
    BL[3] = Dh1 * XT[0][1] + Dh0 * XT[1][1];
    BL[4] = Dh2 * XT[1][1] + Dh1 * XT[2][1];
    BL[5] = Dh2 * XT[0][1] + Dh0 * XT[2][1];
    FY = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /// Computes Z component
    BL[0] = Dh0 * XT[0][2];
    BL[1] = Dh1 * XT[1][2];
    BL[2] = Dh2 * XT[2][2];
    BL[3] = Dh1 * XT[0][2] + Dh0 * XT[1][2];
    BL[4] = Dh2 * XT[1][2] + Dh1 * XT[2][2];
    BL[5] = Dh2 * XT[0][2] + Dh0 * XT[2][2];
    FZ = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    // Writes into global memory
    return make_float4(FX, FY, FZ, 0.0f);

}




/**
 * This kernel gathers the forces by element computed by the first kernel to each node
 */
__global__ void CudaTetrahedronTLEDForceField3f_addForce_kernel(
    int nbVertex, int nbElements, unsigned int valence, float* f, int2* forceCoordinates,
    float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = blockIdx.x * BSIZE; //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    int index3 = index1 * 3; //3*index1;
    int iext = index0 * 3 + index1;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    // Variable used for reading textures
    float4 Read;

    /**
    * Sums nodal forces components
    */
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    int2 FCrds;

    for (int val=0; val<valence; val++)
    {
        // Grabs the force coordinate (slice, texture index)
        int nd = valence*index+val;
        FCrds = forceCoordinates[nd];
        if (FCrds.y < 0 || FCrds.y >= nbElements)
        {
            FCrds.y = 0;
        }

        // Retrieves the force components for that node at that index and on that slice
        switch ( FCrds.x )
        {
        case 0:
            Read = F0[FCrds.y];
            break;

        case 1:
            Read = F1[FCrds.y];
            break;

        case 2:
            Read = F2[FCrds.y];
            break;

        case 3:
            Read = F3[FCrds.y];
            break;

        default:
            Read = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            break;
        }

        // Sums
        Fx += Read.x;
        Fy += Read.y;
        Fz += Read.z;
    }


    temp[index3  ] = Fx;
    temp[index3+1] = Fy;
    temp[index3+2] = Fz;

    __syncthreads();

    f[iext        ] -= temp[index1        ];
    f[iext+  BSIZE] -= temp[index1+  BSIZE];
    f[iext+2*BSIZE] -= temp[index1+2*BSIZE];

}


/*
 * CPU-side methods
 */


/**
 * Initialises GPU textures with the precomputed arrays for the TLED algorithm
 */
void InitGPU_TetrahedronTLED(int valence, int nbVertex, int nbElements)
{
    mycudaPrintf("GPU initialised for TLED: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Initialises GPU textures with the precomputed arrays for the viscoelastic formulation
 */
void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv)
{
    // Constants A and B for isochoric part
    if (Ni !=0)
    {
        cudaMemcpyToSymbol(Ai_gpu, Ai, 2*Ni*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    // Constants A and B for volumetric part
    if (Nv != 0)
    {
        cudaMemcpyToSymbol(Av_gpu, Av, 2*Nv*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    mycudaPrintf("GPU initialised for viscoelasticity: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Initialises GPU textures with the precomputed arrays for the anisotropic formulation
 */
void InitGPU_TetrahedronAniso()
{
    // A material constant
    const int Eta = 13136;    // 13136 liver
    cudaMemcpyToSymbol("Eta_gpu", &Eta, sizeof(int));

    mycudaPrintf("GPU initialised for anisotropy: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the TLED
 */
void ClearGPU_TetrahedronTLED(void)
{
    mycudaPrintf("GPU memory cleaned for TLED: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the viscoelastic formulation
 */
void ClearGPU_TetrahedronVisco(void)
{
    mycudaPrintf("GPU memory cleaned for viscoelasticity: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the anisotropic formulation
 */
void ClearGPU_TetrahedronAniso(void)
{
    mycudaPrintf("GPU memory cleaned for anisotropy: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Calls the two kernels to compute the internal forces
 */
void CudaTetrahedronTLEDForceField3f_addForce(int4* nodesPerElement, float4* DhC0, float4* DhC1,
                                              float4* DhC2, float* volume,
                                              int2* forceCoordinates,
                                              float3* preferredDirection,
                                              float4* Di1, float4* Di2, float4* Dv1, float4* Dv2,
                                              float4* F0, float4* F1, float4* F2, float4* F3,
                                              float Lambda, float Mu,
                                              unsigned nbElem, unsigned nbVertex,
                                              unsigned nbElemPerVertex,
                                              unsigned viscoelasticity, unsigned anisotropy,
                                              const float* x, const float* x0, void* f)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);

    /**
     * The first kernel operates over elements in the model and computes the element stresses based on the current model configuration.
     * It then converts them into nodal force contributions.
     *
     * The right formulation is chosen based on the binary value of ab (with a = boolean for viscoelasticity and b = boolean for anisotropy)
     */
    switch(2*viscoelasticity + anisotropy)
    {
    case 0 :
        // Isotropic
    {CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet0<<< grid1, threads1>>>(x, x0, nodesPerElement, DhC0, DhC1, DhC2, volume, Lambda, Mu, nbElem, F0, F1, F2, F3); mycudaDebugError("CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet0");}
    break;

    case 1 :
        // Anisotropic
    {CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet1<<< grid1, threads1>>>(x, x0, nodesPerElement, DhC0, DhC1, DhC2, volume, preferredDirection, Lambda, Mu, nbElem, F0, F1, F2, F3); mycudaDebugError("CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet1");}
    break;

    case 2 :
        // Viscoelastic
    {CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet2<<< grid1, threads1>>>(x, x0, nodesPerElement, DhC0, DhC1, DhC2, volume, Lambda, Mu, nbElem, Di1, Di2, Dv1, Dv2, F0, F1, F2, F3); mycudaDebugError("CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet2");}
    break;

    case 3 :
        // Viscoelastic and anisotropic
    {CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet3<<< grid1, threads1>>>(x, x0, nodesPerElement, DhC0, DhC1, DhC2, volume, preferredDirection, Lambda, Mu, nbElem, Di1, Di2, Dv1, Dv2, F0, F1, F2, F3); mycudaDebugError("CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet3");}
    break;
    }

    // The second kernel operates over nodes and reads the previously calculated element force contributions and sums them for each node
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaTetrahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElem, nbElemPerVertex, (float*)f, forceCoordinates, F0, F1, F2, F3); mycudaDebugError("CudaTetrahedronTLEDForceField3f_addForce_kernel");}

}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
