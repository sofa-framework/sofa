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
    void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const float* x, const float* x0, void* f, int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2, float* detJarray, float* hourglassControlArray, float3* preferredDirection, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2, int2* forceCoordinates, float4* F0, float4* F1, float4* F2, float4* F3, float4* F4, float4* F5, float4* F6, float4* F7);
    void InitGPU_TLED(int valence, int nbVertex, int nbElements);
    void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv);
    void InitGPU_Aniso();
    void ClearGPU_TLED(void);
    void ClearGPU_Visco(void);
    void ClearGPU_Aniso(void);
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

// A material constant used for the transversely isotropy
static __constant__ int Eta_gpu;

// Function to be called from the device to compute forces from stresses (Prototype)
__device__ float4 computeForce_hex(const int node, const float4 Dh0_a, const float4 Dh0_b, const float4 Dh1_a, const float4 Dh1_b, const float4 Dh2_a, const float4 Dh2_b, const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp, const float3 Node4Disp, const float3 Node5Disp, const float3 Node6Disp, const float3 Node7Disp, const float3 Node8Disp, const float * SPK, const int tid, float* hourglassControlArray);

// Device function returning the position of vertex i as a float3
__device__ CudaVec3f hexaGetX(const float* x, int i)
{
    const int i3 = i * 3;
    const float x1 = x[i3];
    const float x2 = x[i3+1];
    const float x3 = x[i3+2];
    return CudaVec3f::make(x1,x2,x3);
}

/**
 * This version is valid for hexahedral meshes and uses an elastic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel0(
    const float* x, const float* x0,
    float Lambda, float Mu, int nbElem, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu,
    float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2,
    float* detJarray, float* hourglassControlArray)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = DhC0[2 * index];
        float4 Dh0_b = DhC0[2 * index + 1];
        float4 Dh1_a = DhC1[2 * index];
        float4 Dh1_b = DhC1[2 * index + 1];
        float4 Dh2_a = DhC2[2 * index];
        float4 Dh2_b = DhC2[2 * index + 1];

        int4 NodesPerElement = nodesPerElement[2 * index];
        CudaVec3f Node1Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        NodesPerElement = nodesPerElement[2 * index + 1];
        CudaVec3f Node5Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node6Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node7Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node8Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        //Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        //Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        //Column 3
        XT[0][2] =  Dh0_a.x*Node1Disp.z + Dh0_a.y*Node2Disp.z + Dh0_a.z*Node3Disp.z + Dh0_a.w*Node4Disp.z +
                Dh0_b.x*Node5Disp.z + Dh0_b.y*Node6Disp.z + Dh0_b.z*Node7Disp.z + Dh0_b.w*Node8Disp.z;
        XT[1][2] =  Dh1_a.x*Node1Disp.z + Dh1_a.y*Node2Disp.z + Dh1_a.z*Node3Disp.z + Dh1_a.w*Node4Disp.z +
                Dh1_b.x*Node5Disp.z + Dh1_b.y*Node6Disp.z + Dh1_b.z*Node7Disp.z + Dh1_b.w*Node8Disp.z;
        XT[2][2] =  Dh2_a.x*Node1Disp.z + Dh2_a.y*Node2Disp.z + Dh2_a.z*Node3Disp.z + Dh2_a.w*Node4Disp.z +
                Dh2_b.x*Node5Disp.z + Dh2_b.y*Node6Disp.z + Dh2_b.z*Node7Disp.z + Dh2_b.w*Node8Disp.z + 1.0f;


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

        // Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        // C inverse
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        // Isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        float x4 = __fdividef(-x1*(C11+C22+C33), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        // Elastic component of the response (isochoric part + volumetric part)
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

        // Gets the Jacobian determinant
        float detJ = detJarray[index];
        SPK[0] *= 8*detJ;
        SPK[1] *= 8*detJ;
        SPK[2] *= 8*detJ;
        SPK[3] *= 8*detJ;
        SPK[4] *= 8*detJ;
        SPK[5] *= 8*detJ;

        /**
        * Computes strain-displacement matrix
        */
        F0_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);


    }

}

/**
 * This version is valid for hexahedral meshes and uses a transversely isotropic and elastic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel1(
    const float* x, const float* x0,
    float Lambda, float Mu, int nbElem, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu,
    float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2,
    float* detJarray, float* hourglassControlArray,
    float3* preferredDirection)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = DhC0[2 * index];
        float4 Dh0_b = DhC0[2 * index + 1];
        float4 Dh1_a = DhC1[2 * index];
        float4 Dh1_b = DhC1[2 * index + 1];
        float4 Dh2_a = DhC2[2 * index];
        float4 Dh2_b = DhC2[2 * index + 1];

        int4 NodesPerElement = nodesPerElement[2 * index];
        CudaVec3f Node1Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        NodesPerElement = nodesPerElement[2 * index + 1];
        CudaVec3f Node5Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node6Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node7Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node8Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        //Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        //Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        //Column 3
        XT[0][2] =  Dh0_a.x*Node1Disp.z + Dh0_a.y*Node2Disp.z + Dh0_a.z*Node3Disp.z + Dh0_a.w*Node4Disp.z +
                Dh0_b.x*Node5Disp.z + Dh0_b.y*Node6Disp.z + Dh0_b.z*Node7Disp.z + Dh0_b.w*Node8Disp.z;
        XT[1][2] =  Dh1_a.x*Node1Disp.z + Dh1_a.y*Node2Disp.z + Dh1_a.z*Node3Disp.z + Dh1_a.w*Node4Disp.z +
                Dh1_b.x*Node5Disp.z + Dh1_b.y*Node6Disp.z + Dh1_b.z*Node7Disp.z + Dh1_b.w*Node8Disp.z;
        XT[2][2] =  Dh2_a.x*Node1Disp.z + Dh2_a.y*Node2Disp.z + Dh2_a.z*Node3Disp.z + Dh2_a.w*Node4Disp.z +
                Dh2_b.x*Node5Disp.z + Dh2_b.y*Node6Disp.z + Dh2_b.z*Node7Disp.z + Dh2_b.w*Node8Disp.z + 1.0f;


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

        // Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        // C inverse
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        // Transversely isotropic
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

        // Elastic component of the response (isochoric part + volumetric part)
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

        // Gets the Jacobian determinant
        float detJ = detJarray[index];
        SPK[0] *= 8*detJ;
        SPK[1] *= 8*detJ;
        SPK[2] *= 8*detJ;
        SPK[3] *= 8*detJ;
        SPK[4] *= 8*detJ;
        SPK[5] *= 8*detJ;

        /**
        * Computes strain-displacement matrix
        */
        F0_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);


    }

}

/**
 * This version is valid for hexahedral meshes and uses a viscoelastic formulation based on separated isochoric and volumetric terms. The model is isotropic.
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel2(
    const float* x, const float* x0,
    float Lambda, float Mu, int nbElem, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2,
    float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu,
    float4* F6_gpu, float4* F7_gpu,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2,
    float* detJarray, float* hourglassControlArray)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = DhC0[2 * index];
        float4 Dh0_b = DhC0[2 * index + 1];
        float4 Dh1_a = DhC1[2 * index];
        float4 Dh1_b = DhC1[2 * index + 1];
        float4 Dh2_a = DhC2[2 * index];
        float4 Dh2_b = DhC2[2 * index + 1];

        int4 NodesPerElement = nodesPerElement[2 * index];
        CudaVec3f Node1Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        NodesPerElement = nodesPerElement[2 * index + 1];
        CudaVec3f Node5Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node6Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node7Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node8Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        //Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        //Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        //Column 3
        XT[0][2] =  Dh0_a.x*Node1Disp.z + Dh0_a.y*Node2Disp.z + Dh0_a.z*Node3Disp.z + Dh0_a.w*Node4Disp.z +
                Dh0_b.x*Node5Disp.z + Dh0_b.y*Node6Disp.z + Dh0_b.z*Node7Disp.z + Dh0_b.w*Node8Disp.z;
        XT[1][2] =  Dh1_a.x*Node1Disp.z + Dh1_a.y*Node2Disp.z + Dh1_a.z*Node3Disp.z + Dh1_a.w*Node4Disp.z +
                Dh1_b.x*Node5Disp.z + Dh1_b.y*Node6Disp.z + Dh1_b.z*Node7Disp.z + Dh1_b.w*Node8Disp.z;
        XT[2][2] =  Dh2_a.x*Node1Disp.z + Dh2_a.y*Node2Disp.z + Dh2_a.z*Node3Disp.z + Dh2_a.w*Node4Disp.z +
                Dh2_b.x*Node5Disp.z + Dh2_b.y*Node6Disp.z + Dh2_b.z*Node7Disp.z + Dh2_b.w*Node8Disp.z + 1.0f;


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

        // Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        // C inverse
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        // Isotropic
        float J23 = __powf(J, -(float)2/3);   // J23 = J^(-2/3)
        float x1 = J23*Mu;
        float x4 = __fdividef(-x1*(C11+C22+C33), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        // Elastic component of the response (isochoric part + volumetric part)
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


        // Viscoelastic components of response
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
        temp = Dv1[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Dv1[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        // Gets the Jacobian determinant
        float detJ = detJarray[index];
        SPK[0] *= 8*detJ;
        SPK[1] *= 8*detJ;
        SPK[2] *= 8*detJ;
        SPK[3] *= 8*detJ;
        SPK[4] *= 8*detJ;
        SPK[5] *= 8*detJ;

        /**
        * Computes strain-displacement matrix
        */
        F0_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);


    }

}

/**
 * This version is valid for hexahedral meshes and uses an viscoelastic and anisotropic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel3(
    const float* x, const float* x0,
    float Lambda, float Mu, int nbElem, float4* Di1, float4* Di2, float4* Dv1, float4* Dv2,
    float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu,
    float4* F6_gpu, float4* F7_gpu,
    int4* nodesPerElement, float4* DhC0, float4* DhC1, float4* DhC2,
    float* detJarray, float* hourglassControlArray,
    float3* preferredDirection)
{
    int index0 = blockIdx.x * BSIZE;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = DhC0[2 * index];
        float4 Dh0_b = DhC0[2 * index + 1];
        float4 Dh1_a = DhC1[2 * index];
        float4 Dh1_b = DhC1[2 * index + 1];
        float4 Dh2_a = DhC2[2 * index];
        float4 Dh2_b = DhC2[2 * index + 1];

        int4 NodesPerElement = nodesPerElement[2 * index];
        CudaVec3f Node1Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node2Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node3Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node4Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        NodesPerElement = nodesPerElement[2 * index + 1];
        CudaVec3f Node5Disp = hexaGetX(x, NodesPerElement.x) - hexaGetX(x0, NodesPerElement.x);
        CudaVec3f Node6Disp = hexaGetX(x, NodesPerElement.y) - hexaGetX(x0, NodesPerElement.y);
        CudaVec3f Node7Disp = hexaGetX(x, NodesPerElement.z) - hexaGetX(x0, NodesPerElement.z);
        CudaVec3f Node8Disp = hexaGetX(x, NodesPerElement.w) - hexaGetX(x0, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        //Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        //Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        //Column 3
        XT[0][2] =  Dh0_a.x*Node1Disp.z + Dh0_a.y*Node2Disp.z + Dh0_a.z*Node3Disp.z + Dh0_a.w*Node4Disp.z +
                Dh0_b.x*Node5Disp.z + Dh0_b.y*Node6Disp.z + Dh0_b.z*Node7Disp.z + Dh0_b.w*Node8Disp.z;
        XT[1][2] =  Dh1_a.x*Node1Disp.z + Dh1_a.y*Node2Disp.z + Dh1_a.z*Node3Disp.z + Dh1_a.w*Node4Disp.z +
                Dh1_b.x*Node5Disp.z + Dh1_b.y*Node6Disp.z + Dh1_b.z*Node7Disp.z + Dh1_b.w*Node8Disp.z;
        XT[2][2] =  Dh2_a.x*Node1Disp.z + Dh2_a.y*Node2Disp.z + Dh2_a.z*Node3Disp.z + Dh2_a.w*Node4Disp.z +
                Dh2_b.x*Node5Disp.z + Dh2_b.y*Node6Disp.z + Dh2_b.z*Node7Disp.z + Dh2_b.w*Node8Disp.z + 1.0f;


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

        // Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        // C inverse
        float Ci11, Ci12, Ci13, Ci22, Ci23, Ci33;
        Ci11 = (C22*C33 - C23*C23)*invdetC;
        Ci12 = (C13*C23 - C12*C33)*invdetC;
        Ci13 = (C12*C23 - C13*C22)*invdetC;
        Ci22 = (C11*C33 - C13*C13)*invdetC;
        Ci23 = (C12*C13 - C11*C23)*invdetC;
        Ci33 = (C11*C22 - C12*C12)*invdetC;

        // Transversely isotropic
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

        // Elastic component of the response (isochoric part + volumetric part)
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


        // Viscoelastic components of response
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
        temp = Dv1[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[tid] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = Dv2[index];
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[tid] = make_float4(temp.x, temp.y, 0, 0);
#endif

        // Gets the Jacobian determinant
        float detJ = detJarray[index];
        SPK[0] *= 8*detJ;
        SPK[1] *= 8*detJ;
        SPK[2] *= 8*detJ;
        SPK[3] *= 8*detJ;
        SPK[4] *= 8*detJ;
        SPK[5] *= 8*detJ;

        /**
        * Computes strain-displacement matrix
        */
        F0_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index, hourglassControlArray);


    }

}

/**
 * Function to be called from the device to compute forces from stresses
 * => allows us to save registers by recomputing the deformation gradient
 */
__device__ float4 computeForce_hex(const int node, const float4 Dh0_a, const float4 Dh0_b, const float4 Dh1_a,
        const float4 Dh1_b, const float4 Dh2_a, const float4 Dh2_b, const float3 Node1Disp,
        const float3 Node2Disp, const float3 Node3Disp, const float3 Node4Disp,
        const float3 Node5Disp, const float3 Node6Disp, const float3 Node7Disp,
        const float3 Node8Disp, const float * SPK, const int tid,
        float* hourglassControlArray)
{
    float XT[3][3];

    //Column 1
    XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
            Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
    XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
            Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
    XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
            Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

    //Column 2
    XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
            Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
    XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
            Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
    XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
            Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

    //Column 3
    XT[0][2] =  Dh0_a.x*Node1Disp.z + Dh0_a.y*Node2Disp.z + Dh0_a.z*Node3Disp.z + Dh0_a.w*Node4Disp.z +
            Dh0_b.x*Node5Disp.z + Dh0_b.y*Node6Disp.z + Dh0_b.z*Node7Disp.z + Dh0_b.w*Node8Disp.z;
    XT[1][2] =  Dh1_a.x*Node1Disp.z + Dh1_a.y*Node2Disp.z + Dh1_a.z*Node3Disp.z + Dh1_a.w*Node4Disp.z +
            Dh1_b.x*Node5Disp.z + Dh1_b.y*Node6Disp.z + Dh1_b.z*Node7Disp.z + Dh1_b.w*Node8Disp.z;
    XT[2][2] =  Dh2_a.x*Node1Disp.z + Dh2_a.y*Node2Disp.z + Dh2_a.z*Node3Disp.z + Dh2_a.w*Node4Disp.z +
            Dh2_b.x*Node5Disp.z + Dh2_b.y*Node6Disp.z + Dh2_b.z*Node7Disp.z + Dh2_b.w*Node8Disp.z + 1.0f;


    float BL[6];
    float FX, FY, FZ;
    float4 HG_read;

    float Dh0, Dh1, Dh2;
    switch(node)
    {
    case 0 :
        Dh0 = Dh0_a.x;
        Dh1 = Dh1_a.x;
        Dh2 = Dh2_a.x;
        break;

    case 1 :
        Dh0 = Dh0_a.y;
        Dh1 = Dh1_a.y;
        Dh2 = Dh2_a.y;
        break;

    case 2 :
        Dh0 = Dh0_a.z;
        Dh1 = Dh1_a.z;
        Dh2 = Dh2_a.z;
        break;

    case 3 :
        Dh0 = Dh0_a.w;
        Dh1 = Dh1_a.w;
        Dh2 = Dh2_a.w;
        break;

    case 4 :
        Dh0 = Dh0_b.x;
        Dh1 = Dh1_b.x;
        Dh2 = Dh2_b.x;
        break;

    case 5 :
        Dh0 = Dh0_b.y;
        Dh1 = Dh1_b.y;
        Dh2 = Dh2_b.y;
        break;

    case 6 :
        Dh0 = Dh0_b.z;
        Dh1 = Dh1_b.z;
        Dh2 = Dh2_b.z;
        break;

    case 7 :
        Dh0 = Dh0_b.w;
        Dh1 = Dh1_b.w;
        Dh2 = Dh2_b.w;
        break;

    }


    // Computes X component
    BL[0] = Dh0 * XT[0][0];
    BL[1] = Dh1 * XT[1][0];
    BL[2] = Dh2 * XT[2][0];
    BL[3] = Dh1 * XT[0][0] + Dh0 * XT[1][0];
    BL[4] = Dh2 * XT[1][0] + Dh1 * XT[2][0];
    BL[5] = Dh2 * XT[0][0] + Dh0 * XT[2][0];
    FX = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    // Computes Y component
    BL[0] = Dh0 * XT[0][1];
    BL[1] = Dh1 * XT[1][1];
    BL[2] = Dh2 * XT[2][1];
    BL[3] = Dh1 * XT[0][1] + Dh0 * XT[1][1];
    BL[4] = Dh2 * XT[1][1] + Dh1 * XT[2][1];
    BL[5] = Dh2 * XT[0][1] + Dh0 * XT[2][1];
    FY = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    // Computes Z component
    BL[0] = Dh0 * XT[0][2];
    BL[1] = Dh1 * XT[1][2];
    BL[2] = Dh2 * XT[2][2];
    BL[3] = Dh1 * XT[0][2] + Dh0 * XT[1][2];
    BL[4] = Dh2 * XT[1][2] + Dh1 * XT[2][2];
    BL[5] = Dh2 * XT[0][2] + Dh0 * XT[2][2];
    FZ = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /**
     * Computes hourglass control force vector for first node (FHG = HG * u)
     */
    // Computes the hourglass force to add
    HG_read.x = hourglassControlArray[4*(16*tid+2*node)+0];
    HG_read.y = hourglassControlArray[4*(16*tid+2*node)+1];
    HG_read.z = hourglassControlArray[4*(16*tid+2*node)+2];
    HG_read.w = hourglassControlArray[4*(16*tid+2*node)+3];
    FX += HG_read.x*Node1Disp.x + HG_read.y*Node2Disp.x + HG_read.z*Node3Disp.x + HG_read.w*Node4Disp.x;
    FY += HG_read.x*Node1Disp.y + HG_read.y*Node2Disp.y + HG_read.z*Node3Disp.y + HG_read.w*Node4Disp.y;
    FZ += HG_read.x*Node1Disp.z + HG_read.y*Node2Disp.z + HG_read.z*Node3Disp.z + HG_read.w*Node4Disp.z;

    HG_read.x = hourglassControlArray[4*(16*tid+2*node)+4];
    HG_read.y = hourglassControlArray[4*(16*tid+2*node)+5];
    HG_read.z = hourglassControlArray[4*(16*tid+2*node)+6];
    HG_read.w = hourglassControlArray[4*(16*tid+2*node)+7];
    FX += HG_read.x*Node5Disp.x + HG_read.y*Node6Disp.x + HG_read.z*Node7Disp.x + HG_read.w*Node8Disp.x;
    FY += HG_read.x*Node5Disp.y + HG_read.y*Node6Disp.y + HG_read.z*Node7Disp.y + HG_read.w*Node8Disp.y;
    FZ += HG_read.x*Node5Disp.z + HG_read.y*Node6Disp.z + HG_read.z*Node7Disp.z + HG_read.w*Node8Disp.z;

    // Writes into global memory
    return make_float4( FX, FY, FZ, 0);

}


/**
 * This kernel gathers the forces by element computed by the first kernel to each node
 */
__global__ void CudaHexahedronTLEDForceField3f_addForce_kernel(
    int nbVertex, int nbElements, unsigned int valence, float* f/*, float* test*/,
    int2* forceCoordinates,
    float4* F0, float4* F1, float4* F2, float4* F3, float4* F4, float4* F5, float4* F6, float4* F7)
{
    int index0 = blockIdx.x * BSIZE;
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

        case 4:
            Read = F4[FCrds.y];
        break;

        case 5:
            Read = F5[FCrds.y];
        break;

        case 6:
            Read = F6[FCrds.y];
        break;

        case 7:
            Read = F7[FCrds.y];
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


/**
 * CPU-side methods
 */


/**
 * Initialises GPU textures with the precomputed arrays for the TLED algorithm
 */
void InitGPU_TLED(int valence, int nbVertex, int nbElements)
{}

/**
 * Initialises GPU textures with the precomputed arrays for the viscoelastic formulation
 */
void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv)
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
void InitGPU_Aniso()
{
    // A material constant
    const int Eta = 13136;    // 13136 liver
    cudaMemcpyToSymbol("Eta_gpu", &Eta, sizeof(int));

    mycudaPrintf("GPU initialised for anisotropy: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the TLED
 */
void ClearGPU_TLED(void)
{
    mycudaPrintf("GPU memory cleaned for TLED: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_Visco(void)
{}

/**
 * Deletes all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_Aniso(void)
{}

/**
 * Calls the two kernels to compute the internal forces
 */
void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem,
                                             unsigned int nbVertex, unsigned int nbElemPerVertex,
                                             unsigned int viscoelasticity, unsigned int anisotropy,
                                             const float* x, const float* x0, void* f,
                                             int4* nodesPerElement,
                                             float4* DhC0, float4* DhC1, float4* DhC2,
                                             float* detJarray, float* hourglassControlArray,
                                             float3* preferredDirection,
                                             float4* Di1, float4* Di2, float4* Dv1, float4* Dv2,
                                             int2* forceCoordinates,
                                             float4* F0, float4* F1, float4* F2, float4* F3, float4* F4, float4* F5, float4* F6, float4* F7)
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
    {CudaHexahedronTLEDForceField3f_calcForce_kernel0<<< grid1, threads1>>>(x, x0, Lambda, Mu, nbElem, F0, F1, F2, F3, F4, F5, F6, F7, nodesPerElement, DhC0, DhC1, DhC2, detJarray, hourglassControlArray); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel0");}
    break;

    case 1 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel1<<< grid1, threads1>>>(x, x0, Lambda, Mu, nbElem, F0, F1, F2, F3, F4, F5, F6, F7, nodesPerElement, DhC0, DhC1, DhC2, detJarray, hourglassControlArray, preferredDirection); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel1");}
    break;

    case 2 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel2<<< grid1, threads1>>>(x, x0, Lambda, Mu, nbElem, Di1, Di2, Dv1, Dv2, F0, F1, F2, F3, F4, F5, F6, F7, nodesPerElement, DhC0, DhC1, DhC2, detJarray, hourglassControlArray); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel2");}
    break;

    case 3 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel3<<< grid1, threads1>>>(x, x0, Lambda, Mu, nbElem, Di1, Di2, Dv1, Dv2, F0, F1, F2, F3, F4, F5, F6, F7, nodesPerElement, DhC0, DhC1, DhC2, detJarray, hourglassControlArray, preferredDirection); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel3");}
    break;
    }

    // The second kernel operates over nodes and reads the previously calculated element force contributions and sums them for each node
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaHexahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElem, nbElemPerVertex, (float*)f, forceCoordinates, F0, F1, F2, F3, F4, F5, F6, F7); mycudaDebugError("CudaHexahedronTLEDForceField3f_addForce_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
