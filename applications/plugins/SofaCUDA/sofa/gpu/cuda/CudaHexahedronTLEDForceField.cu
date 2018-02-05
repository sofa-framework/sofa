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
    void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f);
    void InitGPU_TLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* DetJ, float* HG, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv);
    void InitGPU_Aniso(float* A);
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

// References on textures - TLED first kernel
static texture <int4, 1, cudaReadModeElementType> texNodesPerElement;
static texture <float4, 1, cudaReadModeElementType> texDhC0;
static texture <float4, 1, cudaReadModeElementType> texDhC1;
static texture <float4, 1, cudaReadModeElementType> texDhC2;
static texture <float, 1, cudaReadModeElementType> texDetJ;
static texture <float4, 1, cudaReadModeElementType> texDisp;

// Hourglass control
static texture <float4, 1, cudaReadModeElementType> texHG;

// Constant used with anisotropic formulations
static texture <float4, 1, cudaReadModeElementType> texA;

// Viscoelasticity
static texture <float4, 1, cudaReadModeElementType> texDi1;
static texture <float4, 1, cudaReadModeElementType> texDi2;
static texture <float4, 1, cudaReadModeElementType> texDv1;
static texture <float4, 1, cudaReadModeElementType> texDv2;

// References on textures - TLED second kernel
static texture <int2, 1, cudaReadModeElementType> texFCrds;
static texture <float4, 1, cudaReadModeElementType> texF0;
static texture <float4, 1, cudaReadModeElementType> texF1;
static texture <float4, 1, cudaReadModeElementType> texF2;
static texture <float4, 1, cudaReadModeElementType> texF3;
static texture <float4, 1, cudaReadModeElementType> texF4;
static texture <float4, 1, cudaReadModeElementType> texF5;
static texture <float4, 1, cudaReadModeElementType> texF6;
static texture <float4, 1, cudaReadModeElementType> texF7;

/**
 * GPU pointers
 */
// List of nodes for each element
static int4* NodesPerElement_gpu = 0;
// Shape function derivatives arrays
static float4* DhC0_gpu = 0;
static float4* DhC1_gpu = 0;
static float4* DhC2_gpu = 0;
// Hourglass control
static float4* HG_gpu = 0;
// Force coordinates for each node
static int2* FCrds_gpu = 0;
// Jacobian determinant array
float* DetJ_gpu;
// Element nodal force contribution
static float4* F0_gpu = 0;
static float4* F1_gpu = 0;
static float4* F2_gpu = 0;
static float4* F3_gpu = 0;
static float4* F4_gpu = 0;
static float4* F5_gpu = 0;
static float4* F6_gpu = 0;
static float4* F7_gpu = 0;
// Displacements
static float4* Disp = 0;

// Array that contains the preferred direction for each element (transverse isotropy)
static float4* A_gpu = 0;

// Viscoelasticity
static float4 * Di1_gpu = 0;
static float4 * Di2_gpu = 0;
static float4 * Dv1_gpu = 0;
static float4 * Dv2_gpu = 0;



// Function to be called from the device to compute forces from stresses (Prototype)
__device__ float4 computeForce_hex(const int node, const float4 Dh0_a, const float4 Dh0_b, const float4 Dh1_a, const float4 Dh1_b, const float4 Dh2_a, const float4 Dh2_b, const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp, const float3 Node4Disp, const float3 Node5Disp, const float3 Node6Disp, const float3 Node7Disp, const float3 Node8Disp, const float * SPK, const int tid);

// A few global constants
static int sizeNodesInt, sizeElsFloat, sizeElsInt;

#define USE_TEXTURE

#ifdef USE_TEXTURE

static texture<float, 1, cudaReadModeElementType> texX;
static const void* curX = NULL;

static texture<float, 1, cudaReadModeElementType> texX0;
static const void* curX0 = NULL;

static void setX(const void* x)
{
    if (x!=curX)
    {
        cudaBindTexture((size_t*)NULL, texX, x);
        curX = x;
    }
}

// Device function returning the position of vertex i as a float3
static __device__ CudaVec3f getX(int i)
{
    int i3 = umul24(i,3);
    float x1 = tex1Dfetch(texX, i3);
    float x2 = tex1Dfetch(texX, i3+1);
    float x3 = tex1Dfetch(texX, i3+2);
    return CudaVec3f::make(x1,x2,x3);
}

static void setX0(const void* x0)
{
    if (x0!=curX0)
    {
        cudaBindTexture((size_t*)NULL, texX0, x0);
        curX0 = x0;
    }
}

// Device function returning the rest position of vertex i as a float3
static __device__ CudaVec3f getX0(int i)
{
    int i3 = umul24(i,3);
    float x1 = tex1Dfetch(texX0, i3);
    float x2 = tex1Dfetch(texX0, i3+1);
    float x3 = tex1Dfetch(texX0, i3+2);
    return CudaVec3f::make(x1,x2,x3);
}

#else

static void setX(const void* x)
{
}

static void setX0(const void* x0)
{
}

#define getX(i) (((const CudaVec3f*)x)[i])
#define getX0(i) (((const CudaVec3f*)x0)[i])

#endif

/**
 * This version is valid for hexahedral meshes and uses an elastic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel0(float Lambda, float Mu, int nbElem, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = tex1Dfetch(texDhC0, 2*index);
        float4 Dh0_b = tex1Dfetch(texDhC0, 2*index+1);
        float4 Dh1_a = tex1Dfetch(texDhC1, 2*index);
        float4 Dh1_b = tex1Dfetch(texDhC1, 2*index+1);
        float4 Dh2_a = tex1Dfetch(texDhC2, 2*index);
        float4 Dh2_b = tex1Dfetch(texDhC2, 2*index+1);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index);
        CudaVec3f Node1Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node2Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node3Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node4Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

        NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
        CudaVec3f Node5Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node6Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node7Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node8Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

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
        float detJ = tex1Dfetch(texDetJ, index);
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
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

/**
 * This version is valid for hexahedral meshes and uses a transversely isotropic and elastic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel1(float Lambda, float Mu, int nbElem, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = tex1Dfetch(texDhC0, 2*index);
        float4 Dh0_b = tex1Dfetch(texDhC0, 2*index+1);
        float4 Dh1_a = tex1Dfetch(texDhC1, 2*index);
        float4 Dh1_b = tex1Dfetch(texDhC1, 2*index+1);
        float4 Dh2_a = tex1Dfetch(texDhC2, 2*index);
        float4 Dh2_b = tex1Dfetch(texDhC2, 2*index+1);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index);
        CudaVec3f Node1Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node2Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node3Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node4Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

        NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
        CudaVec3f Node5Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node6Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node7Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node8Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

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
        float4 a = tex1Dfetch(texA, index);

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
        float detJ = tex1Dfetch(texDetJ, index);
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
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

/**
 * This version is valid for hexahedral meshes and uses a viscoelastic formulation based on separated isochoric and volumetric terms. The model is isotropic.
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel2(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = tex1Dfetch(texDhC0, 2*index);
        float4 Dh0_b = tex1Dfetch(texDhC0, 2*index+1);
        float4 Dh1_a = tex1Dfetch(texDhC1, 2*index);
        float4 Dh1_b = tex1Dfetch(texDhC1, 2*index+1);
        float4 Dh2_a = tex1Dfetch(texDhC2, 2*index);
        float4 Dh2_b = tex1Dfetch(texDhC2, 2*index+1);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index);
        CudaVec3f Node1Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node2Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node3Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node4Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

        NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
        CudaVec3f Node5Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node6Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node7Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node8Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

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
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_gpu[1]; temp.z += Ai_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_gpu[+1]; temp.w += Ai_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(texDv1, index);
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDv2, index);
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        // Gets the Jacobian determinant
        float detJ = tex1Dfetch(texDetJ, index);
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
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

/**
 * This version is valid for hexahedral meshes and uses an viscoelastic and anisotropic formulation
 */
__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel3(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0_gpu, float4* F1_gpu, float4* F2_gpu, float4* F3_gpu, float4* F4_gpu, float4* F5_gpu, float4* F6_gpu, float4* F7_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        // Shape function derivatives matrix
        float4 Dh0_a = tex1Dfetch(texDhC0, 2*index);
        float4 Dh0_b = tex1Dfetch(texDhC0, 2*index+1);
        float4 Dh1_a = tex1Dfetch(texDhC1, 2*index);
        float4 Dh1_b = tex1Dfetch(texDhC1, 2*index+1);
        float4 Dh2_a = tex1Dfetch(texDhC2, 2*index);
        float4 Dh2_b = tex1Dfetch(texDhC2, 2*index+1);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index);
        CudaVec3f Node1Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node2Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node3Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node4Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

        NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
        CudaVec3f Node5Disp = getX(NodesPerElement.x) - getX0(NodesPerElement.x);
        CudaVec3f Node6Disp = getX(NodesPerElement.y) - getX0(NodesPerElement.y);
        CudaVec3f Node7Disp = getX(NodesPerElement.z) - getX0(NodesPerElement.z);
        CudaVec3f Node8Disp = getX(NodesPerElement.w) - getX0(NodesPerElement.w);

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
        float4 a = tex1Dfetch(texA, index);

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
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_gpu[1]; temp.z += Ai_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_gpu[+1]; temp.w += Ai_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Ai_gpu[1]; temp.x += Ai_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_gpu[1]; temp.y += Ai_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(texDv1, index);
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_gpu[1]; temp.z += Av_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_gpu[1]; temp.w += Av_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[tid] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDv2, index);
        temp.x *= Av_gpu[1]; temp.x += Av_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_gpu[1]; temp.y += Av_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[tid] = make_float4(temp.x, temp.y, 0, 0);
#endif

        // Gets the Jacobian determinant
        float detJ = tex1Dfetch(texDetJ, index);
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
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


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
        const float3 Node8Disp, const float * SPK, const int tid)
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
    HG_read = tex1Dfetch(texHG, 16*tid+2*node);
    FX += HG_read.x*Node1Disp.x + HG_read.y*Node2Disp.x + HG_read.z*Node3Disp.x + HG_read.w*Node4Disp.x;
    FY += HG_read.x*Node1Disp.y + HG_read.y*Node2Disp.y + HG_read.z*Node3Disp.y + HG_read.w*Node4Disp.y;
    FZ += HG_read.x*Node1Disp.z + HG_read.y*Node2Disp.z + HG_read.z*Node3Disp.z + HG_read.w*Node4Disp.z;

    HG_read = tex1Dfetch(texHG, 16*tid+2*node+1);
    FX += HG_read.x*Node5Disp.x + HG_read.y*Node6Disp.x + HG_read.z*Node7Disp.x + HG_read.w*Node8Disp.x;
    FY += HG_read.x*Node5Disp.y + HG_read.y*Node6Disp.y + HG_read.z*Node7Disp.y + HG_read.w*Node8Disp.y;
    FZ += HG_read.x*Node5Disp.z + HG_read.y*Node6Disp.z + HG_read.z*Node7Disp.z + HG_read.w*Node8Disp.z;

    // Writes into global memory
    return make_float4( FX, FY, FZ, 0);

}


/**
 * This kernel gathers the forces by element computed by the first kernel to each node
 */
__global__ void CudaHexahedronTLEDForceField3f_addForce_kernel(int nbVertex, unsigned int valence, float* f/*, float* test*/)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    int index3 = umul24(index1,3); //3*index1;
    int iext = umul24(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

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
        FCrds = tex1Dfetch(texFCrds, nd);


        // Retrieves the force components for that node at that index and on that slice
        switch ( FCrds.x )
        {
        case 0:
            Read = tex1Dfetch(texF0, FCrds.y);
            break;

        case 1:
            Read = tex1Dfetch(texF1, FCrds.y);
            break;

        case 2:
            Read = tex1Dfetch(texF2, FCrds.y);
            break;

        case 3:
            Read = tex1Dfetch(texF3, FCrds.y);
            break;

        case 4:
            Read = tex1Dfetch(texF4, FCrds.y);
            break;

        case 5:
            Read = tex1Dfetch(texF5, FCrds.y);
            break;

        case 6:
            Read = tex1Dfetch(texF6, FCrds.y);
            break;

        case 7:
            Read = tex1Dfetch(texF7, FCrds.y);
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
void InitGPU_TLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* DetJ, float* HG, int* FCrds, int valence, int nbVertex, int nbElements)
{
    // Sizes in bytes of different arrays
    sizeNodesInt = nbVertex*sizeof(int);
    sizeElsFloat = nbElements*sizeof(float);
    sizeElsInt = nbElements*sizeof(int);

    // List of nodes for each element
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&NodesPerElement_gpu, 8*sizeElsInt);
    mycudaMemcpyHostToDevice(NodesPerElement_gpu, NodesPerElement, 8*sizeElsInt);
    cudaBindTexture(0, texNodesPerElement, NodesPerElement_gpu, channelDesc);

    // First shape function derivatives array (first column for each element)
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&DhC0_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC0_gpu, DhC0, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC0, DhC0_gpu, channelDesc);

    // Second shape function derivatives array (second column for each element)
    mycudaMalloc((void**)&DhC1_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC1_gpu, DhC1, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC1, DhC1_gpu, channelDesc);

    // Third shape function derivatives array (third column for each element)
    mycudaMalloc((void**)&DhC2_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC2_gpu, DhC2, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC2, DhC2_gpu, channelDesc);

    // Hourglass control
    mycudaMalloc((void**)&HG_gpu, 64*sizeElsFloat);
    mycudaMemcpyHostToDevice(HG_gpu, HG, 64*sizeElsFloat);
    cudaBindTexture(0, texHG, HG_gpu, channelDesc);


    // Jacobian determinant array
    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&DetJ_gpu, sizeElsFloat);
    mycudaMemcpyHostToDevice(DetJ_gpu, DetJ, sizeElsFloat);
    cudaBindTexture(0, texDetJ, DetJ_gpu, channelDesc);


    /**
     * Allocates force arrays and zeros them
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&F0_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF0, F0_gpu, channelDesc);

    mycudaMalloc((void**)&F1_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF1, F1_gpu, channelDesc);

    mycudaMalloc((void**)&F2_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF2, F2_gpu, channelDesc);

    mycudaMalloc((void**)&F3_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF3, F3_gpu, channelDesc);

    mycudaMalloc((void**)&F4_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF4, F4_gpu, channelDesc);

    mycudaMalloc((void**)&F5_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF5, F5_gpu, channelDesc);

    mycudaMalloc((void**)&F6_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF6, F6_gpu, channelDesc);

    mycudaMalloc((void**)&F7_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF7, F7_gpu, channelDesc);


    /**
     * Displacements array
     */
    mycudaMalloc((void**)&Disp, 4*BSIZE*(int)ceil((float)nbVertex/BSIZE)*sizeof(float));
    cudaBindTexture(0, texDisp, Disp, channelDesc);

    /**
     * Force coordinates array
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&FCrds_gpu, 2*sizeNodesInt*valence);
    mycudaMemcpyHostToDevice(FCrds_gpu, FCrds, 2*sizeNodesInt*valence);
    cudaBindTexture(0, texFCrds, FCrds_gpu, channelDesc);

    mycudaPrintf("GPU initialised for TLED: %s\n", cudaGetErrorString( cudaGetLastError()) );

}

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

    // Rate-dependant stress (isochoric part)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    if (Ni != 0)
    {
        mycudaMalloc((void**)&Di1_gpu, 4*sizeElsFloat);
        cudaMemset(Di1_gpu, 0, 4*sizeElsFloat);
        cudaBindTexture(0, texDi1, Di1_gpu, channelDesc);

        mycudaMalloc((void**)&Di2_gpu, 4*sizeElsFloat);
        cudaMemset(Di2_gpu, 0, 4*sizeElsFloat);
        cudaBindTexture(0, texDi2, Di2_gpu, channelDesc);
    }

    // Rate-dependant stress (volumetric part)
    if (Nv != 0)
    {
        mycudaMalloc((void**)&Dv1_gpu, 4*sizeElsFloat);
        cudaMemset(Dv1_gpu, 0, 4*sizeElsFloat);
        cudaBindTexture(0, texDv1, Dv1_gpu, channelDesc);

        mycudaMalloc((void**)&Dv2_gpu, 4*sizeElsFloat);
        cudaMemset(Dv2_gpu, 0, 4*sizeElsFloat);
        cudaBindTexture(0, texDv2, Dv2_gpu, channelDesc);
    }

    mycudaPrintf("GPU initialised for viscoelasticity: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Initialises GPU textures with the precomputed arrays for the anisotropic formulation
 */
void InitGPU_Aniso(float* A)
{
    // A material constant
    int Eta = 13136;    // 13136 liver
    cudaMemcpyToSymbol("Eta_gpu", &Eta, sizeof(int));

    // Preferred direction for each element (transverse isotropy)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&A_gpu, 4*sizeElsFloat);
    cudaMemcpy((void*)A_gpu, (void*)A, 4*sizeElsFloat, cudaMemcpyHostToDevice);
    cudaBindTexture(0, texA, A_gpu, channelDesc);

    mycudaPrintf("GPU initialised for anisotropy: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the TLED
 */
void ClearGPU_TLED(void)
{
    mycudaFree(NodesPerElement_gpu);
    mycudaFree(DhC0_gpu);
    mycudaFree(DhC1_gpu);
    mycudaFree(DhC2_gpu);
    mycudaFree(HG_gpu);
    mycudaFree(DetJ_gpu);
    mycudaFree(FCrds_gpu);
    mycudaFree(F0_gpu);
    mycudaFree(F1_gpu);
    mycudaFree(F2_gpu);
    mycudaFree(F3_gpu);
    mycudaFree(F4_gpu);
    mycudaFree(F5_gpu);
    mycudaFree(F6_gpu);
    mycudaFree(F7_gpu);

    mycudaFree(Disp);

    mycudaPrintf("GPU memory cleaned for TLED: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_Visco(void)
{
    mycudaFree(Di1_gpu);
    mycudaFree(Di2_gpu);
    mycudaFree(Dv1_gpu);
    mycudaFree(Dv2_gpu);

    mycudaPrintf("GPU memory cleaned for viscoelasticity: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Deletes all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_Aniso(void)
{
    mycudaFree(A_gpu);

    mycudaPrintf("GPU memory cleaned for anisotropy: %s\n", cudaGetErrorString( cudaGetLastError()) );
}

/**
 * Calls the two kernels to compute the internal forces
 */
void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f)
{
    setX(x);
    setX0(x0);

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
    {CudaHexahedronTLEDForceField3f_calcForce_kernel0<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0_gpu, F1_gpu, F2_gpu, F3_gpu, F4_gpu, F5_gpu, F6_gpu, F7_gpu); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel0");}
    break;

    case 1 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel1<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0_gpu, F1_gpu, F2_gpu, F3_gpu, F4_gpu, F5_gpu, F6_gpu, F7_gpu); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel1");}
    break;

    case 2 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel2<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1_gpu, Di2_gpu, Dv1_gpu, Dv2_gpu, F0_gpu, F1_gpu, F2_gpu, F3_gpu, F4_gpu, F5_gpu, F6_gpu, F7_gpu); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel2");}
    break;

    case 3 :
    {CudaHexahedronTLEDForceField3f_calcForce_kernel3<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1_gpu, Di2_gpu, Dv1_gpu, Dv2_gpu, F0_gpu, F1_gpu, F2_gpu, F3_gpu, F4_gpu, F5_gpu, F6_gpu, F7_gpu); mycudaDebugError("CudaHexahedronTLEDForceField3f_calcForce_kernel3");}
    break;
    }

    // The second kernel operates over nodes and reads the previously calculated element force contributions and sums them for each node
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    {CudaHexahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (float*)f); mycudaDebugError("CudaHexahedronTLEDForceField3f_addForce_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
