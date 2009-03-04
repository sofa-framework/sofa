/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
    void CudaHexahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
    void InitGPU_TLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* DetJ, float* HG, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv, int nbElements);
    void InitGPU_Aniso(void);
    void ClearGPU_TLED(void);
    void ClearGPU_Visco(void);
}

class __align__(16) GPUElement
{
public:
    /// @name index of the 8 connected vertices
    /// @{
    int v[8];
    /// @}
};

class __align__(16) GPUElementState
{
public:
    int dummy[8];

};

//////////////////////
// GPU-side methods //
//////////////////////

#define ISOCHORIC

/// Constant memory
// A few constants for viscoelasticity
__constant__ float Ai_hex_gpu[2];
__constant__ float Av_hex_gpu[2];

// A few constants used for the transversely isotropy
__constant__ int Eta_hex_gpu;       // Material constant
__constant__ float A_hex_gpu[6];    // Structure tensor defining the preferred material direction

/// References on textures
// TLED first kernel
texture <int4, 1, cudaReadModeElementType> texNodesPerElement;
texture <float4, 1, cudaReadModeElementType> texDhC0;
texture <float4, 1, cudaReadModeElementType> texDhC1;
texture <float4, 1, cudaReadModeElementType> texDhC2;
texture <float, 1, cudaReadModeElementType> texDetJ;
texture <float4, 1, cudaReadModeElementType> texDisp;

// Hourglass control
texture <float4, 1, cudaReadModeElementType> texHG;

// Viscoelasticity
texture <float4, 1, cudaReadModeElementType> Di1_ref;
texture <float4, 1, cudaReadModeElementType> Di2_ref;
texture <float4, 1, cudaReadModeElementType> Dv1_ref;
texture <float4, 1, cudaReadModeElementType> Dv2_ref;

// TLED second kernel
texture <int2, 1, cudaReadModeElementType> texFCrds;
texture <float4, 1, cudaReadModeElementType> texF0;
texture <float4, 1, cudaReadModeElementType> texF1;
texture <float4, 1, cudaReadModeElementType> texF2;
texture <float4, 1, cudaReadModeElementType> texF3;
texture <float4, 1, cudaReadModeElementType> texF4;
texture <float4, 1, cudaReadModeElementType> texF5;
texture <float4, 1, cudaReadModeElementType> texF6;
texture <float4, 1, cudaReadModeElementType> texF7;

/// GPU pointers
// List of nodes for each element
int4* NodesPerElement_hex_gpu = 0;
// Shape function derivatives arrays
float4* DhC0_hex_gpu = 0;
float4* DhC1_hex_gpu = 0;
float4* DhC2_hex_gpu = 0;
// Hourglass control
float4* HG_hex_gpu = 0;
// Force coordinates for each node
int2* FCrds_hex_gpu = 0;
// Jacobian determinant array
float* DetJ_hex_gpu;
// Element nodal force contribution
float4* F0_hex_gpu = 0;
float4* F1_hex_gpu = 0;
float4* F2_hex_gpu = 0;
float4* F3_hex_gpu = 0;
float4* F4_hex_gpu = 0;
float4* F5_hex_gpu = 0;
float4* F6_hex_gpu = 0;
float4* F7_hex_gpu = 0;
// Displacements
float4* Disp = 0;

// Viscoelasticity
float4 * Di1_hex_gpu = 0;
float4 * Di2_hex_gpu = 0;
float4 * Dv1_hex_gpu = 0;
float4 * Dv2_hex_gpu = 0;

/// CPU pointers
float* test = 0;
float* force = 0;
float* testCPU = 0;

/** Protopypes
 */

// Function to be called from the device to compute forces from stresses
// => allows to save registers by recomputing the deformation gradient
__device__ float4 computeForce_hex(const int node, const float4 Dh0_a, const float4 Dh0_b, const float4 Dh1_a, const float4 Dh1_b, const float4 Dh2_a, const float4 Dh2_b, const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp, const float3 Node4Disp, const float3 Node5Disp, const float3 Node6Disp, const float3 Node7Disp, const float3 Node8Disp, const float * SPK, const int tid);

__device__ float4 getDisp(const int NodeID);

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

__device__ CudaVec3f getX(int i)
{
    int i3 = umul24(i,3);
    float x1 = tex1Dfetch(texX, i3);
    float x2 = tex1Dfetch(texX, i3+1);
    float x3 = tex1Dfetch(texX, i3+2);
    return CudaVec3f::make(x1,x2,x3);
}

// __device__ float4 getX(int i)
// {
//     return tex1Dfetch(texX, i);
// }

static void setX0(const void* x0)
{
    if (x0!=curX0)
    {
        cudaBindTexture((size_t*)NULL, texX0, x0);
        curX0 = x0;
    }
}

__device__ CudaVec3f getX0(int i)
{
    int i3 = umul24(i,3);
    float x1 = tex1Dfetch(texX0, i3);
    float x2 = tex1Dfetch(texX0, i3+1);
    float x3 = tex1Dfetch(texX0, i3+2);
    return CudaVec3f::make(x1,x2,x3);
}

// __device__ float4 getX0(int i)
// {
//     return tex1Dfetch(texX0, i);
// }

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

__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel0(float Lambda, float Mu, int nbElem, float4* F0_hex_gpu, float4* F1_hex_gpu, float4* F2_hex_gpu, float4* F3_hex_gpu, float4* F4_hex_gpu, float4* F5_hex_gpu, float4* F6_hex_gpu, float4* F7_hex_gpu/*, float* test*/)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
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

//         float4 Node1Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node2Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node3Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node4Disp = tex1Dfetch(texDisp, NodesPerElement.w);
//
//         NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
//         float4 Node5Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node6Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node7Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node8Disp = tex1Dfetch(texDisp, NodesPerElement.w);


        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        ///Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        ///Column 3
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

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverse
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

        /// Get the Jacobian determinant
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
        F0_hex_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_hex_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_hex_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_hex_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_hex_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_hex_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_hex_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_hex_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel1(float Lambda, float Mu, int nbElem, float4* F0_hex_gpu, float4* F1_hex_gpu, float4* F2_hex_gpu, float4* F3_hex_gpu, float4* F4_hex_gpu, float4* F5_hex_gpu, float4* F6_hex_gpu, float4* F7_hex_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
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

//         float4 Node1Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node2Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node3Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node4Disp = tex1Dfetch(texDisp, NodesPerElement.w);
//
//         NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
//         float4 Node5Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node6Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node7Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node8Disp = tex1Dfetch(texDisp, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        ///Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        ///Column 3
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

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverse
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
        float x2 = J23*(A_hex_gpu[0]*C11+A_hex_gpu[1]*C22+A_hex_gpu[2]*C33+2*A_hex_gpu[3]*C12+2*A_hex_gpu[4]*C23+2*A_hex_gpu[5]*C13) - 1;
        float x3 = J23*Eta_hex_gpu*x2;
        float x4 = __fdividef(-(Eta_hex_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*A_hex_gpu[0] + x4*Ci11 + x1;
        SiE22 = x3*A_hex_gpu[1] + x4*Ci22 + x1;
        SiE33 = x3*A_hex_gpu[2] + x4*Ci33 + x1;
        SiE12 = x3*A_hex_gpu[3] + x4*Ci12;
        SiE23 = x3*A_hex_gpu[4] + x4*Ci23;
        SiE13 = x3*A_hex_gpu[5] + x4*Ci13;

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

        /// Get the Jacobian determinant
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
        F0_hex_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_hex_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_hex_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_hex_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_hex_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_hex_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_hex_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_hex_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel2(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0_hex_gpu, float4* F1_hex_gpu, float4* F2_hex_gpu, float4* F3_hex_gpu, float4* F4_hex_gpu, float4* F5_hex_gpu, float4* F6_hex_gpu, float4* F7_hex_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
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

//         float4 Node1Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node2Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node3Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node4Disp = tex1Dfetch(texDisp, NodesPerElement.w);
//
//         NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
//         float4 Node5Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node6Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node7Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node8Disp = tex1Dfetch(texDisp, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        ///Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        ///Column 3
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

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverse
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
        temp = tex1Dfetch(Di1_ref, index);
        temp.x *= Ai_hex_gpu[1]; temp.x += Ai_hex_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_hex_gpu[1]; temp.y += Ai_hex_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_hex_gpu[1]; temp.z += Ai_hex_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_hex_gpu[+1]; temp.w += Ai_hex_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(Di2_ref, index);
        temp.x *= Ai_hex_gpu[1]; temp.x += Ai_hex_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_hex_gpu[1]; temp.y += Ai_hex_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(Dv1_ref, index);
        temp.x *= Av_hex_gpu[1]; temp.x += Av_hex_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_hex_gpu[1]; temp.y += Av_hex_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_hex_gpu[1]; temp.z += Av_hex_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_hex_gpu[1]; temp.w += Av_hex_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(Dv2_ref, index);
        temp.x *= Av_hex_gpu[1]; temp.x += Av_hex_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_hex_gpu[1]; temp.y += Av_hex_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Get the Jacobian determinant
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
        F0_hex_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_hex_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_hex_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_hex_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_hex_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_hex_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_hex_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_hex_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}

__global__ void CudaHexahedronTLEDForceField3f_calcForce_kernel3(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0_hex_gpu, float4* F1_hex_gpu, float4* F2_hex_gpu, float4* F3_hex_gpu, float4* F4_hex_gpu, float4* F5_hex_gpu, float4* F6_hex_gpu, float4* F7_hex_gpu)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
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

//         float4 Node1Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node2Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node3Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node4Disp = tex1Dfetch(texDisp, NodesPerElement.w);
//
//         NodesPerElement = tex1Dfetch(texNodesPerElement, 2*index+1);
//         float4 Node5Disp = tex1Dfetch(texDisp, NodesPerElement.x);
//         float4 Node6Disp = tex1Dfetch(texDisp, NodesPerElement.y);
//         float4 Node7Disp = tex1Dfetch(texDisp, NodesPerElement.z);
//         float4 Node8Disp = tex1Dfetch(texDisp, NodesPerElement.w);

        /**
        * Computes the transpose of deformation gradient
        *
        * Transpose of displacement derivatives = transpose(shape function derivatives) * ElementNodalDisplacement
        * Transpose of deformation gradient = transpose of displacement derivatives + identity
        */
        float XT[3][3];

        ///Column 1
        XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
                Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
        XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
                Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
        XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
                Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

        ///Column 2
        XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
                Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
        XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
                Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
        XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
                Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

        ///Column 3
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

        /// Determinant of C
        float invdetC = __fdividef(1.0f, C11*(C22*C33 - C23*C23)
                - C12*(C12*C33 - C23*C13)
                + C13*(C12*C23 - C22*C13) );

        /// C inverse
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
        float x2 = J23*(A_hex_gpu[0]*C11+A_hex_gpu[1]*C22+A_hex_gpu[2]*C33+2*A_hex_gpu[3]*C12+2*A_hex_gpu[4]*C23+2*A_hex_gpu[5]*C13) - 1;
        float x3 = J23*Eta_hex_gpu*x2;
        float x4 = __fdividef(-(Eta_hex_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*A_hex_gpu[0] + x4*Ci11 + x1;
        SiE22 = x3*A_hex_gpu[1] + x4*Ci22 + x1;
        SiE33 = x3*A_hex_gpu[2] + x4*Ci33 + x1;
        SiE12 = x3*A_hex_gpu[3] + x4*Ci12;
        SiE23 = x3*A_hex_gpu[4] + x4*Ci23;
        SiE13 = x3*A_hex_gpu[5] + x4*Ci13;

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
        temp = tex1Dfetch(Di1_ref, index);
        temp.x *= Ai_hex_gpu[1]; temp.x += Ai_hex_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_hex_gpu[1]; temp.y += Ai_hex_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_hex_gpu[1]; temp.z += Ai_hex_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_hex_gpu[+1]; temp.w += Ai_hex_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(Di2_ref, index);
        temp.x *= Ai_hex_gpu[1]; temp.x += Ai_hex_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_hex_gpu[1]; temp.y += Ai_hex_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(Dv1_ref, index);
        temp.x *= Av_hex_gpu[1]; temp.x += Av_hex_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_hex_gpu[1]; temp.y += Av_hex_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_hex_gpu[1]; temp.z += Av_hex_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_hex_gpu[1]; temp.w += Av_hex_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[tid] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(Dv2_ref, index);
        temp.x *= Av_hex_gpu[1]; temp.x += Av_hex_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_hex_gpu[1]; temp.y += Av_hex_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[tid] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Get the Jacobian determinant
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
        F0_hex_gpu[index] = computeForce_hex(0, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F1_hex_gpu[index] = computeForce_hex(1, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F2_hex_gpu[index] = computeForce_hex(2, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F3_hex_gpu[index] = computeForce_hex(3, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F4_hex_gpu[index] = computeForce_hex(4, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F5_hex_gpu[index] = computeForce_hex(5, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F6_hex_gpu[index] = computeForce_hex(6, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);

        F7_hex_gpu[index] = computeForce_hex(7, Dh0_a, Dh0_b, Dh1_a, Dh1_b, Dh2_a, Dh2_b, Node1Disp, Node2Disp, Node3Disp, Node4Disp,
                Node5Disp, Node6Disp, Node7Disp, Node8Disp, SPK, index);


    }

}


/**
 * Computes a float4 displacement from float3s current and initial positions
 */
__global__ void calcDeplacement3d_to_4d_kernel(const float* x, const float* x0, float4* disp)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index1 = threadIdx.x;
    int index = index0+index1;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];
    temp[index1] = x[index]-x0[index];
    temp[index1+BSIZE] = x[index+BSIZE]-x0[index+BSIZE];
    temp[index1+2*BSIZE] = x[index+2*BSIZE]-x0[index+2*BSIZE];

    __syncthreads();

    float4 r;
    int index3 = umul24(index1,3);
    r.x = temp[index3+0];
    r.y = temp[index3+1];
    r.z = temp[index3+2];

    disp[umul24(blockIdx.x,BSIZE*3)+index1] = r;

}

/**
 * Function to be called from the device to compute forces from stresses
 * => allows to save registers by recomputing the deformation gradient (NEED TO CHECK THIS AGAIN)
 */
__device__ float4 computeForce_hex(const int node, const float4 Dh0_a, const float4 Dh0_b, const float4 Dh1_a,
        const float4 Dh1_b, const float4 Dh2_a, const float4 Dh2_b, const float3 Node1Disp,
        const float3 Node2Disp, const float3 Node3Disp, const float3 Node4Disp,
        const float3 Node5Disp, const float3 Node6Disp, const float3 Node7Disp,
        const float3 Node8Disp, const float * SPK, const int tid)
{
    float XT[3][3];

    ///Column 1
    XT[0][0] =  Dh0_a.x*Node1Disp.x + Dh0_a.y*Node2Disp.x + Dh0_a.z*Node3Disp.x + Dh0_a.w*Node4Disp.x +
            Dh0_b.x*Node5Disp.x + Dh0_b.y*Node6Disp.x + Dh0_b.z*Node7Disp.x + Dh0_b.w*Node8Disp.x + 1.0f;
    XT[1][0] =  Dh1_a.x*Node1Disp.x + Dh1_a.y*Node2Disp.x + Dh1_a.z*Node3Disp.x + Dh1_a.w*Node4Disp.x +
            Dh1_b.x*Node5Disp.x + Dh1_b.y*Node6Disp.x + Dh1_b.z*Node7Disp.x + Dh1_b.w*Node8Disp.x;
    XT[2][0] =  Dh2_a.x*Node1Disp.x + Dh2_a.y*Node2Disp.x + Dh2_a.z*Node3Disp.x + Dh2_a.w*Node4Disp.x +
            Dh2_b.x*Node5Disp.x + Dh2_b.y*Node6Disp.x + Dh2_b.z*Node7Disp.x + Dh2_b.w*Node8Disp.x;

    ///Column 2
    XT[0][1] =  Dh0_a.x*Node1Disp.y + Dh0_a.y*Node2Disp.y + Dh0_a.z*Node3Disp.y + Dh0_a.w*Node4Disp.y +
            Dh0_b.x*Node5Disp.y + Dh0_b.y*Node6Disp.y + Dh0_b.z*Node7Disp.y + Dh0_b.w*Node8Disp.y;
    XT[1][1] =  Dh1_a.x*Node1Disp.y + Dh1_a.y*Node2Disp.y + Dh1_a.z*Node3Disp.y + Dh1_a.w*Node4Disp.y +
            Dh1_b.x*Node5Disp.y + Dh1_b.y*Node6Disp.y + Dh1_b.z*Node7Disp.y + Dh1_b.w*Node8Disp.y + 1.0f;
    XT[2][1] =  Dh2_a.x*Node1Disp.y + Dh2_a.y*Node2Disp.y + Dh2_a.z*Node3Disp.y + Dh2_a.w*Node4Disp.y +
            Dh2_b.x*Node5Disp.y + Dh2_b.y*Node6Disp.y + Dh2_b.z*Node7Disp.y + Dh2_b.w*Node8Disp.y;

    ///Column 3
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


    /// Compute X component
    BL[0] = Dh0 * XT[0][0];
    BL[1] = Dh1 * XT[1][0];
    BL[2] = Dh2 * XT[2][0];
    BL[3] = Dh1 * XT[0][0] + Dh0 * XT[1][0];
    BL[4] = Dh2 * XT[1][0] + Dh1 * XT[2][0];
    BL[5] = Dh2 * XT[0][0] + Dh0 * XT[2][0];
    FX = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /// Compute Y component
    BL[0] = Dh0 * XT[0][1];
    BL[1] = Dh1 * XT[1][1];
    BL[2] = Dh2 * XT[2][1];
    BL[3] = Dh1 * XT[0][1] + Dh0 * XT[1][1];
    BL[4] = Dh2 * XT[1][1] + Dh1 * XT[2][1];
    BL[5] = Dh2 * XT[0][1] + Dh0 * XT[2][1];
    FY = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /// Compute Z component
    BL[0] = Dh0 * XT[0][2];
    BL[1] = Dh1 * XT[1][2];
    BL[2] = Dh2 * XT[2][2];
    BL[3] = Dh1 * XT[0][2] + Dh0 * XT[1][2];
    BL[4] = Dh2 * XT[1][2] + Dh1 * XT[2][2];
    BL[5] = Dh2 * XT[0][2] + Dh0 * XT[2][2];
    FZ = SPK[0]*BL[0] + SPK[1]*BL[1] + SPK[2]*BL[2] + SPK[3]*BL[3] + SPK[4]*BL[4] + SPK[5]*BL[5];

    /**
     * Compute hourglass control force vector for first node (FHG = HG * u)
     */
    // Compute the hourglass force to add
    HG_read = tex1Dfetch(texHG, 16*tid+2*node);
    FX += HG_read.x*Node1Disp.x + HG_read.y*Node2Disp.x + HG_read.z*Node3Disp.x + HG_read.w*Node4Disp.x;
    FY += HG_read.x*Node1Disp.y + HG_read.y*Node2Disp.y + HG_read.z*Node3Disp.y + HG_read.w*Node4Disp.y;
    FZ += HG_read.x*Node1Disp.z + HG_read.y*Node2Disp.z + HG_read.z*Node3Disp.z + HG_read.w*Node4Disp.z;

    HG_read = tex1Dfetch(texHG, 16*tid+2*node+1);
    FX += HG_read.x*Node5Disp.x + HG_read.y*Node6Disp.x + HG_read.z*Node7Disp.x + HG_read.w*Node8Disp.x;
    FY += HG_read.x*Node5Disp.y + HG_read.y*Node6Disp.y + HG_read.z*Node7Disp.y + HG_read.w*Node8Disp.y;
    FZ += HG_read.x*Node5Disp.z + HG_read.y*Node6Disp.z + HG_read.z*Node7Disp.z + HG_read.w*Node8Disp.z;

    // Write in global memory
    return make_float4( FX, FY, FZ, 0);

}


__global__ void CudaHexahedronTLEDForceField3f_addForce_kernel(int nbVertex, unsigned int valence, float* f/*, float* test*/)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
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


__global__ void CudaHexahedronTLEDForceField3f_calcDForce_kernel(int nbElem, const GPUElement* elems, GPUElementState* state, const float* x)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    //GPUElement e = elems[index];
    GPUElementState s = state[index];

    if (index < nbElem)
    {
    }

    state[index] = s;

}

//////////////////////
// CPU-side methods //
//////////////////////


/** Initialise GPU textures with the precomputed arrays for the TLED algorithm
 */
void InitGPU_TLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* DetJ, float* HG, int* FCrds, int valence, int nbVertex, int nbElements)
{
    /// Sizes in bytes of different arrays
//     int sizeNodesFloat = nbVertex*sizeof(float);
    int sizeNodesInt = nbVertex*sizeof(int);
    int sizeElsFloat = nbElements*sizeof(float);
    int sizeElsInt = nbElements*sizeof(int);

    /// List of nodes for each element
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&NodesPerElement_hex_gpu, 8*sizeElsInt);
    mycudaMemcpyHostToDevice(NodesPerElement_hex_gpu, NodesPerElement, 8*sizeElsInt);
    cudaBindTexture(0, texNodesPerElement, NodesPerElement_hex_gpu, channelDesc);

    /// First shape function derivatives array (first column for each element)
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&DhC0_hex_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC0_hex_gpu, DhC0, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC0, DhC0_hex_gpu, channelDesc);

    /// Second shape function derivatives array (second column for each element)
    mycudaMalloc((void**)&DhC1_hex_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC1_hex_gpu, DhC1, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC1, DhC1_hex_gpu, channelDesc);

    /// Third shape function derivatives array (third column for each element)
    mycudaMalloc((void**)&DhC2_hex_gpu, 8*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC2_hex_gpu, DhC2, 8*sizeElsFloat);
    cudaBindTexture(0, texDhC2, DhC2_hex_gpu, channelDesc);

    /// Hourglass control
    mycudaMalloc((void**)&HG_hex_gpu, 64*sizeElsFloat);
    mycudaMemcpyHostToDevice(HG_hex_gpu, HG, 64*sizeElsFloat);
    cudaBindTexture(0, texHG, HG_hex_gpu, channelDesc);


    /// Jacobian determinant array
    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&DetJ_hex_gpu, sizeElsFloat);
    mycudaMemcpyHostToDevice(DetJ_hex_gpu, DetJ, sizeElsFloat);
    cudaBindTexture(0, texDetJ, DetJ_hex_gpu, channelDesc);


    /**
     * Allocates force arrays and zeros them
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&F0_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF0, F0_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F1_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF1, F1_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F2_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF2, F2_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F3_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF3, F3_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F4_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF4, F4_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F5_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF5, F5_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F6_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF6, F6_hex_gpu, channelDesc);

    mycudaMalloc((void**)&F7_hex_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF7, F7_hex_gpu, channelDesc);


    /**
     * Displacements array
     */
    mycudaMalloc((void**)&Disp, 4*BSIZE*(int)ceil((float)nbVertex/BSIZE)*sizeof(float));
    cudaBindTexture(0, texDisp, Disp, channelDesc);

    /**
     * Force coordinates array
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&FCrds_hex_gpu, 2*sizeNodesInt*valence);
    mycudaMemcpyHostToDevice(FCrds_hex_gpu, FCrds, 2*sizeNodesInt*valence);
    cudaBindTexture(0, texFCrds, FCrds_hex_gpu, channelDesc);


    /**
     * A few arrays used as inputs or outputs by kernels
     */
//     mycudaMalloc((void**)&test, 6*sizeElsFloat);
//     mycudaMalloc((void**)&force, 6*sizeNodesFloat);
//     testCPU = (float*) malloc(6*sizeNodesFloat);

    myprintf("Initialisation GPU for TLED succeeded\n");

}

/** Initialise GPU textures with the precomputed arrays for the viscoelastic formulation
 */
void InitGPU_Visco(float * Ai, float * Av, int Ni, int Nv, int nbElements)
{
    /// Constants A and B for isochoric part
    if (Ni !=0)
    {
        cudaMemcpyToSymbol(Ai_hex_gpu, Ai, 2*Ni*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    /// Constants A and B for volumetric part
    if (Nv != 0)
    {
        cudaMemcpyToSymbol(Av_hex_gpu, Av, 2*Nv*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    /// Rate-dependant stress (isochoric part)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    if (Ni != 0)
    {
        mycudaMalloc((void**)&Di1_hex_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Di1_hex_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, Di1_ref, Di1_hex_gpu, channelDesc);

        mycudaMalloc((void**)&Di2_hex_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Di2_hex_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, Di2_ref, Di2_hex_gpu, channelDesc);
    }

    /// Rate-dependant stress (volumetric part)
    if (Nv != 0)
    {
        mycudaMalloc((void**)&Dv1_hex_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Dv1_hex_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, Dv1_ref, Dv1_hex_gpu, channelDesc);

        mycudaMalloc((void**)&Dv2_hex_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Dv2_hex_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, Dv2_ref, Dv2_hex_gpu, channelDesc);
    }

    myprintf("Initialisation GPU for viscoelasticity succeeded\n");
}

/** Initialise GPU textures with the precomputed arrays for the anisotropic formulation
 */
void InitGPU_Aniso(void)
{
    // A material constant
    int Eta = 13136;    // 13136 liver
    cudaMemcpyToSymbol("Eta_hex_gpu", &Eta, sizeof(int));

    // The structure tensor (a defines the preferred material direction)
    float a[3] = {0, 0.707f, 0.707f};
//     float a[3] = {0.0f, 1.0f, 0.0f};
    float A[6];
    A[0] = a[0]*a[0];    // A(1,1)
    A[1] = a[1]*a[1];    // A(2,2)
    A[2] = a[2]*a[2];    // A(3,3)
    A[3] = a[0]*a[1];    // A(1,2)
    A[4] = a[1]*a[2];    // A(2,3)
    A[5] = a[0]*a[2];    // A(1,3)
    cudaMemcpyToSymbol(A_hex_gpu, A, 6*sizeof(float), 0, cudaMemcpyHostToDevice);

    myprintf("Initialisation GPU for anisotropy succeeded\n");
}

/** Delete all the precomputed arrays allocated for the TLED
 */
void ClearGPU_TLED(void)
{
    mycudaFree(NodesPerElement_hex_gpu);
    mycudaFree(DhC0_hex_gpu);
    mycudaFree(DhC1_hex_gpu);
    mycudaFree(DhC2_hex_gpu);
    mycudaFree(HG_hex_gpu);
    mycudaFree(DetJ_hex_gpu);
    mycudaFree(FCrds_hex_gpu);
    mycudaFree(F0_hex_gpu);
    mycudaFree(F1_hex_gpu);
    mycudaFree(F2_hex_gpu);
    mycudaFree(F3_hex_gpu);
    mycudaFree(F4_hex_gpu);
    mycudaFree(F5_hex_gpu);
    mycudaFree(F6_hex_gpu);
    mycudaFree(F7_hex_gpu);

    mycudaFree(Disp);

    myprintf("Memory on GPU for TLED cleared\n");
}

/** Delete all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_Visco(void)
{
    mycudaFree(Di1_hex_gpu);
    mycudaFree(Di2_hex_gpu);
    mycudaFree(Dv1_hex_gpu);
    mycudaFree(Dv2_hex_gpu);

    myprintf("Memory on GPU for viscoelasticity cleared\n");
}

void CudaHexahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f)
{
    setX(x);
    setX0(x0);

//     dim3 threads3(BSIZE,1);
//     dim3 grid3((nbVertex+BSIZE-1)/BSIZE,1);
//     calcDeplacement3d_to_4d_kernel<<< grid3, threads3, BSIZE*3*sizeof(float) >>>((float*)x, (float*)x0, Disp);

    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);

    /** Pick the right formulation based on the binary value of ab (with a = boolean for viscoelasticity and b = boolean for anisotropy)
     */
    switch(2*viscoelasticity + anisotropy)
    {
    case 0 :
        CudaHexahedronTLEDForceField3f_calcForce_kernel0<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0_hex_gpu, F1_hex_gpu, F2_hex_gpu, F3_hex_gpu, F4_hex_gpu, F5_hex_gpu, F6_hex_gpu, F7_hex_gpu/*, test*/);
        break;

    case 1 :
        CudaHexahedronTLEDForceField3f_calcForce_kernel1<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0_hex_gpu, F1_hex_gpu, F2_hex_gpu, F3_hex_gpu, F4_hex_gpu, F5_hex_gpu, F6_hex_gpu, F7_hex_gpu);
        break;

    case 2 :
        CudaHexahedronTLEDForceField3f_calcForce_kernel2<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1_hex_gpu, Di2_hex_gpu, Dv1_hex_gpu, Dv2_hex_gpu, F0_hex_gpu, F1_hex_gpu, F2_hex_gpu, F3_hex_gpu, F4_hex_gpu, F5_hex_gpu, F6_hex_gpu, F7_hex_gpu);
        break;

    case 3 :
        CudaHexahedronTLEDForceField3f_calcForce_kernel3<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1_hex_gpu, Di2_hex_gpu, Dv1_hex_gpu, Dv2_hex_gpu, F0_hex_gpu, F1_hex_gpu, F2_hex_gpu, F3_hex_gpu, F4_hex_gpu, F5_hex_gpu, F6_hex_gpu, F7_hex_gpu);
        break;
    }

    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    CudaHexahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (float*)f/*, force*/);

    cudaThreadSynchronize();

//     mycudaMemcpyDeviceToHost(testCPU, test, 6*nbElem*sizeof(float));
//     for (int i = 0; i < nbElem; i++)
//     {
//         myprintf("test%d:    %f     %f     %f     %f     %f     %f\n", i, testCPU[6*i], testCPU[6*i+1], testCPU[6*i+2],
//                 testCPU[6*i+3], testCPU[6*i+4], testCPU[6*i+5]);
//     }
//     myprintf("\n");

//     mycudaMemcpyDeviceToHost(testCPU, force, 6*nbVertex*sizeof(float));
//     for (int i = 0; i < nbVertex; i++)
//     {
//         myprintf("force%d:    %f     %f     %f     %f     %f     %f\n", i, testCPU[6*i], testCPU[6*i+1], testCPU[6*i+2],
//                 testCPU[6*i+3], testCPU[6*i+4], testCPU[6*i+5]);
//     }
//     myprintf("\n");
}

void CudaHexahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    //CudaHexahedronTLEDForceField3f_calcDForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)dx);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    //CudaHexahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)df, (const float*)dx);
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
