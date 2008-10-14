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
#include "mycuda.h"

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void CudaTetrahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f);
    void CudaTetrahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
    void InitGPU_TetrahedronTLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* Volume, int* FCrds, int valence, int nbVertex, int nbElements);
    void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv, int nbElements);
    void InitGPU_TetrahedronAniso(void);
    void ClearGPU_TetrahedronTLED(void);
    void ClearGPU_TetrahedronVisco(void);
}

class __align__(16) GPUElement
{
public:
    /// @name index of the 8 connected vertices
    /// @{
    int v[4];
    /// @}
};

class __align__(16) GPUElementState
{
public:
    int dummy[4];

};

//////////////////////
// GPU-side methods //
//////////////////////

#define ISOCHORIC

/// Constant memory
// A few constants for viscoelasticity
__constant__ float Ai_tet_gpu[2];
__constant__ float Av_tet_gpu[2];

// A few constants used for the transversely isotropy
__constant__ int Eta_tet_gpu;       // Material constant
__constant__ float A_tet_gpu[6];    // Structure tensor defining the preferred material direction

/// References on textures
// TLED first kernel
texture <int4, 1, cudaReadModeElementType> texNodesPerElement;
texture <float4, 1, cudaReadModeElementType> texDhC0;
texture <float4, 1, cudaReadModeElementType> texDhC1;
texture <float4, 1, cudaReadModeElementType> texDhC2;
texture <float, 1, cudaReadModeElementType> texVolume;

// Viscoelasticity
texture <float4, 1, cudaReadModeElementType> texDi1;
texture <float4, 1, cudaReadModeElementType> texDi2;
texture <float4, 1, cudaReadModeElementType> texDv1;
texture <float4, 1, cudaReadModeElementType> texDv2;

// TLED second kernel
texture <int2, 1, cudaReadModeElementType> texFCrds;
texture <float4, 1, cudaReadModeElementType> texF0;
texture <float4, 1, cudaReadModeElementType> texF1;
texture <float4, 1, cudaReadModeElementType> texF2;
texture <float4, 1, cudaReadModeElementType> texF3;

/// GPU pointers
// List of nodes for each element
int4* NodesPerElementtet_gpu = 0;
// Shape function derivatives arrays
float4* DhC0tet_gpu = 0;
float4* DhC1tet_gpu = 0;
float4* DhC2tet_gpu = 0;
// Force coordinates for each node
int2* FCrdstet_gpu = 0;
// Element volume array
float* Volumetet_gpu;
// Element nodal force contribution
float4* F0tet_gpu = 0;
float4* F1tet_gpu = 0;
float4* F2tet_gpu = 0;
float4* F3tet_gpu = 0;

// Viscoelasticity
float4 * Di1tet_gpu = 0;
float4 * Di2tet_gpu = 0;
float4 * Dv1tet_gpu = 0;
float4 * Dv2tet_gpu = 0;

/// CPU pointers
// float* blabla = 0;
// float* force = 0;
// float* blablaCPU = 0;

/// Prototype
__device__ float4 computeForce_tet(const int node, const float4 DhC0, const float4 DhC1, const float4 DhC2,
        const float3 Node1Disp, const float3 Node2Disp, const float3 Node3Disp,
        const float3 Node4Disp, const float * SPK, const int tid);

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

__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet0(float Lambda, float Mu, int nbElem, float4* F0, float4* F1, float4* F2, float4* F3/*, float* blabla*/)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
        float4 Dh0 = tex1Dfetch(texDhC0, index);
        float4 Dh1 = tex1Dfetch(texDhC1, index);
        float4 Dh2 = tex1Dfetch(texDhC2, index);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, index);
        CudaVec3f Node1Disp = getX0(NodesPerElement.x) - getX(NodesPerElement.x);
        CudaVec3f Node2Disp = getX0(NodesPerElement.y) - getX(NodesPerElement.y);
        CudaVec3f Node3Disp = getX0(NodesPerElement.z) - getX(NodesPerElement.z);
        CudaVec3f Node4Disp = getX0(NodesPerElement.w) - getX(NodesPerElement.w);

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
        float Vol = tex1Dfetch(texVolume, index);
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

__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet1(float Lambda, float Mu, int nbElem, float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {
        /// Shape function derivatives matrix
        float4 Dh0 = tex1Dfetch(texDhC0, index);
        float4 Dh1 = tex1Dfetch(texDhC1, index);
        float4 Dh2 = tex1Dfetch(texDhC2, index);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, index);
        CudaVec3f Node1Disp = getX0(NodesPerElement.x) - getX(NodesPerElement.x);
        CudaVec3f Node2Disp = getX0(NodesPerElement.y) - getX(NodesPerElement.y);
        CudaVec3f Node3Disp = getX0(NodesPerElement.z) - getX(NodesPerElement.z);
        CudaVec3f Node4Disp = getX0(NodesPerElement.w) - getX(NodesPerElement.w);

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
        float x2 = J23*(A_tet_gpu[0]*C11+A_tet_gpu[1]*C22+A_tet_gpu[2]*C33+2*A_tet_gpu[3]*C12+2*A_tet_gpu[4]*C23+2*A_tet_gpu[5]*C13) - 1;
        float x3 = J23*Eta_tet_gpu*x2;
        float x4 = __fdividef(-(Eta_tet_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*A_tet_gpu[0] + x4*Ci11 + x1;
        SiE22 = x3*A_tet_gpu[1] + x4*Ci22 + x1;
        SiE33 = x3*A_tet_gpu[2] + x4*Ci33 + x1;
        SiE12 = x3*A_tet_gpu[3] + x4*Ci12;
        SiE23 = x3*A_tet_gpu[4] + x4*Ci23;
        SiE13 = x3*A_tet_gpu[5] + x4*Ci13;

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
        float Vol = tex1Dfetch(texVolume, index);
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

__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet2(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {

        /// Shape function derivatives matrix
        float4 Dh0 = tex1Dfetch(texDhC0, index);
        float4 Dh1 = tex1Dfetch(texDhC1, index);
        float4 Dh2 = tex1Dfetch(texDhC2, index);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, index);
        CudaVec3f Node1Disp = getX0(NodesPerElement.x) - getX(NodesPerElement.x);
        CudaVec3f Node2Disp = getX0(NodesPerElement.y) - getX(NodesPerElement.y);
        CudaVec3f Node3Disp = getX0(NodesPerElement.z) - getX(NodesPerElement.z);
        CudaVec3f Node4Disp = getX0(NodesPerElement.w) - getX(NodesPerElement.w);

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
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Ai_tet_gpu[1]; temp.x += Ai_tet_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_tet_gpu[1]; temp.y += Ai_tet_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_tet_gpu[1]; temp.z += Ai_tet_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_tet_gpu[+1]; temp.w += Ai_tet_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Ai_tet_gpu[1]; temp.x += Ai_tet_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_tet_gpu[1]; temp.y += Ai_tet_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Av_tet_gpu[1]; temp.x += Av_tet_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_tet_gpu[1]; temp.y += Av_tet_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_tet_gpu[1]; temp.z += Av_tet_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_tet_gpu[1]; temp.w += Av_tet_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Av_tet_gpu[1]; temp.x += Av_tet_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_tet_gpu[1]; temp.y += Av_tet_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Retrieves the volume
        float Vol = tex1Dfetch(texVolume, index);
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

__global__ void CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet3(float Lambda, float Mu, int nbElem, float4 * Di1, float4 * Di2, float4 * Dv1, float4 * Dv2, float4* F0, float4* F1, float4* F2, float4* F3)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;

    if (index < nbElem)
    {

        /// Shape function derivatives matrix
        float4 Dh0 = tex1Dfetch(texDhC0, index);
        float4 Dh1 = tex1Dfetch(texDhC1, index);
        float4 Dh2 = tex1Dfetch(texDhC2, index);

        int4 NodesPerElement = tex1Dfetch(texNodesPerElement, index);
        CudaVec3f Node1Disp = getX0(NodesPerElement.x) - getX(NodesPerElement.x);
        CudaVec3f Node2Disp = getX0(NodesPerElement.y) - getX(NodesPerElement.y);
        CudaVec3f Node3Disp = getX0(NodesPerElement.z) - getX(NodesPerElement.z);
        CudaVec3f Node4Disp = getX0(NodesPerElement.w) - getX(NodesPerElement.w);

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
        float x2 = J23*(A_tet_gpu[0]*C11+A_tet_gpu[1]*C22+A_tet_gpu[2]*C33+2*A_tet_gpu[3]*C12+2*A_tet_gpu[4]*C23+2*A_tet_gpu[5]*C13) - 1;
        float x3 = J23*Eta_tet_gpu*x2;
        float x4 = __fdividef(-(Eta_tet_gpu*x2*(x2+1)+ x1*(C11+C22+C33)), 3.0f);
        float K = Lambda + __fdividef(2*Mu, 3.0f);
        float x5 = K*J*(J-1);

        /// Elastic component of the response (isochoric part + volumetric part)
        float SiE11, SiE12, SiE13, SiE22, SiE23, SiE33;
        SiE11 = x3*A_tet_gpu[0] + x4*Ci11 + x1;
        SiE22 = x3*A_tet_gpu[1] + x4*Ci22 + x1;
        SiE33 = x3*A_tet_gpu[2] + x4*Ci33 + x1;
        SiE12 = x3*A_tet_gpu[3] + x4*Ci12;
        SiE23 = x3*A_tet_gpu[4] + x4*Ci23;
        SiE13 = x3*A_tet_gpu[5] + x4*Ci13;

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
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Ai_tet_gpu[1]; temp.x += Ai_tet_gpu[0]*SiE11;
        SPK[0] -= temp.x;
        temp.y *= Ai_tet_gpu[1]; temp.y += Ai_tet_gpu[0]*SiE22;
        SPK[1] -= temp.y;
        temp.z *= Ai_tet_gpu[1]; temp.z += Ai_tet_gpu[0]*SiE33;
        SPK[2] -= temp.z;
        temp.w *= Ai_tet_gpu[+1]; temp.w += Ai_tet_gpu[0]*SiE12;
        SPK[3] -= temp.w;
        Di1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Ai_tet_gpu[1]; temp.x += Ai_tet_gpu[0]*SiE23;
        SPK[4] -= temp.x;
        temp.y *= Ai_tet_gpu[1]; temp.y += Ai_tet_gpu[0]*SiE13;
        SPK[5] -= temp.y;
        Di2[index] = make_float4(temp.x, temp.y, 0, 0);

#else
        // Volumetric part
        temp = tex1Dfetch(texDi1, index);
        temp.x *= Av_tet_gpu[1]; temp.x += Av_tet_gpu[0]*SvE11;
        SPK[0] -= temp.x;
        temp.y *= Av_tet_gpu[1]; temp.y += Av_tet_gpu[0]*SvE22;
        SPK[1] -= temp.y;
        temp.z *= Av_tet_gpu[1]; temp.z += Av_tet_gpu[0]*SvE33;
        SPK[2] -= temp.z;
        temp.w *= Av_tet_gpu[1]; temp.w += Av_tet_gpu[0]*SvE12;
        SPK[3] -= temp.w;
        Dv1[index] = make_float4(temp.x, temp.y, temp.z, temp.w);

        temp = tex1Dfetch(texDi2, index);
        temp.x *= Av_tet_gpu[1]; temp.x += Av_tet_gpu[0]*SvE23;
        SPK[4] -= temp.x;
        temp.y *= Av_tet_gpu[1]; temp.y += Av_tet_gpu[0]*SvE13;
        SPK[5] -= temp.y;
        Dv2[index] = make_float4(temp.x, temp.y, 0, 0);
#endif

        /// Retrieves the volume
        float Vol = tex1Dfetch(texVolume, index);
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

    // Write in global memory
    return make_float4( FX, FY, FZ, 0);

}





__global__ void CudaTetrahedronTLEDForceField3f_addForce_kernel(int nbVertex, unsigned int valence, float* f/*, float* test*/)
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

    f[iext        ] += temp[index1        ];
    f[iext+  BSIZE] += temp[index1+  BSIZE];
    f[iext+2*BSIZE] += temp[index1+2*BSIZE];

}


__global__ void CudaTetrahedronTLEDForceField3f_calcDForce_kernel(int nbElem, const GPUElement* elems, GPUElementState* state, const float* x)
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
void InitGPU_TetrahedronTLED(int* NodesPerElement, float* DhC0, float* DhC1, float* DhC2, float* Volume, int* FCrds, int valence, int nbVertex, int nbElements)
{
    /// Sizes in bytes of different arrays
//     int sizeNodesFloat = nbVertex*sizeof(float);
    int sizeNodesInt = nbVertex*sizeof(int);
    int sizeElsFloat = nbElements*sizeof(float);
    int sizeElsInt = nbElements*sizeof(int);

    /// List of nodes for each element
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&NodesPerElementtet_gpu, 4*sizeElsInt);
    mycudaMemcpyHostToDevice(NodesPerElementtet_gpu, NodesPerElement, 4*sizeElsInt);
    cudaBindTexture(0, texNodesPerElement, NodesPerElementtet_gpu, channelDesc);

    /// First shape function derivatives array (first column for each element)
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&DhC0tet_gpu, 4*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC0tet_gpu, DhC0, 4*sizeElsFloat);
    cudaBindTexture(0, texDhC0, DhC0tet_gpu, channelDesc);

    /// Second shape function derivatives array (second column for each element)
    mycudaMalloc((void**)&DhC1tet_gpu, 4*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC1tet_gpu, DhC1, 4*sizeElsFloat);
    cudaBindTexture(0, texDhC1, DhC1tet_gpu, channelDesc);

    /// Third shape function derivatives array (third column for each element)
    mycudaMalloc((void**)&DhC2tet_gpu, 4*sizeElsFloat);
    mycudaMemcpyHostToDevice(DhC2tet_gpu, DhC2, 4*sizeElsFloat);
    cudaBindTexture(0, texDhC2, DhC2tet_gpu, channelDesc);

    /// Jacobian determinant array
    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&Volumetet_gpu, sizeElsFloat);
    mycudaMemcpyHostToDevice(Volumetet_gpu, Volume, sizeElsFloat);
    cudaBindTexture(0, texVolume, Volumetet_gpu, channelDesc);

    /**
     * Allocates force arrays and zeros them
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    mycudaMalloc((void**)&F0tet_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF0, F0tet_gpu, channelDesc);

    mycudaMalloc((void**)&F1tet_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF1, F1tet_gpu, channelDesc);

    mycudaMalloc((void**)&F2tet_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF2, F2tet_gpu, channelDesc);

    mycudaMalloc((void**)&F3tet_gpu, 4*sizeElsFloat);
    cudaBindTexture(0, texF3, F3tet_gpu, channelDesc);

    /**
     * Force coordinates array
     */
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
    mycudaMalloc((void**)&FCrdstet_gpu, 2*sizeNodesInt*valence);
    mycudaMemcpyHostToDevice(FCrdstet_gpu, FCrds, 2*sizeNodesInt*valence);
    cudaBindTexture(0, texFCrds, FCrdstet_gpu, channelDesc);


    /**
     * A few arrays used as inputs or outputs by kernels
     */
//     mycudaMalloc((void**)&blabla, 6*sizeElsFloat);
//     mycudaMalloc((void**)&force, 6*sizeNodesFloat);
//     blablaCPU = (float*) malloc(6*sizeElsFloat);

    myprintf("Initialisation GPU for TLED succeeded\n");

}

/** Initialise GPU textures with the precomputed arrays for the viscoelastic formulation
 */
void InitGPU_TetrahedronVisco(float * Ai, float * Av, int Ni, int Nv, int nbElements)
{
    /// Constants A and B for isochoric part
    if (Ni !=0)
    {
        cudaMemcpyToSymbol(Ai_tet_gpu, Ai, 2*Ni*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    /// Constants A and B for volumetric part
    if (Nv != 0)
    {
        cudaMemcpyToSymbol(Av_tet_gpu, Av, 2*Nv*sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    /// Rate-dependant stress (isochoric part)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    if (Ni != 0)
    {
        mycudaMalloc((void**)&Di1tet_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Di1tet_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, texDi1, Di1tet_gpu, channelDesc);

        mycudaMalloc((void**)&Di2tet_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Di2tet_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, texDi2, Di2tet_gpu, channelDesc);
    }

    /// Rate-dependant stress (volumetric part)
    if (Nv != 0)
    {
        mycudaMalloc((void**)&Dv1tet_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Dv1tet_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, texDv1, Dv1tet_gpu, channelDesc);

        mycudaMalloc((void**)&Dv2tet_gpu, 4*nbElements*sizeof(float));
        cudaMemset(Dv2tet_gpu, 0, 4*nbElements*sizeof(float));
        cudaBindTexture(0, texDv2, Dv2tet_gpu, channelDesc);
    }

    myprintf("Initialisation GPU for viscoelasticity succeeded\n");
}

/** Initialise GPU textures with the precomputed arrays for the anisotropic formulation
 */
void InitGPU_TetrahedronAniso(void)
{
    // A material constant
    int Eta = 13136;    // 13136 liver
    cudaMemcpyToSymbol("Etatet_gpu", &Eta, sizeof(int));

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
    cudaMemcpyToSymbol(A_tet_gpu, A, 6*sizeof(float), 0, cudaMemcpyHostToDevice);

    myprintf("Initialisation GPU for anisotropy succeeded\n");
}

/** Delete all the precomputed arrays allocated for the TLED
 */
void ClearGPU_TetrahedronTLED(void)
{
    mycudaFree(NodesPerElementtet_gpu);
    mycudaFree(DhC0tet_gpu);
    mycudaFree(DhC1tet_gpu);
    mycudaFree(DhC2tet_gpu);
    mycudaFree(Volumetet_gpu);
    mycudaFree(FCrdstet_gpu);
    mycudaFree(F0tet_gpu);
    mycudaFree(F1tet_gpu);
    mycudaFree(F2tet_gpu);
    mycudaFree(F3tet_gpu);

    myprintf("Memory on GPU for TLED cleared\n");
}

/** Delete all the precomputed arrays allocated for the viscoelasticity formulation
 */
void ClearGPU_TetrahedronVisco(void)
{
    mycudaFree(Di1tet_gpu);
    mycudaFree(Di2tet_gpu);
    mycudaFree(Dv1tet_gpu);
    mycudaFree(Dv2tet_gpu);

    myprintf("Memory on GPU for viscoelasticity cleared\n");
}

void CudaTetrahedronTLEDForceField3f_addForce(float Lambda, float Mu, unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, unsigned int viscoelasticity, unsigned int anisotropy, const void* x, const void* x0, void* f)
{
    setX(x);
    setX0(x0);

    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);

    /** Pick the right formulation based on the binary value of ab (with a = boolean for viscoelasticity and b = boolean for anisotropy)
     */
    switch(2*viscoelasticity + anisotropy)
    {
    case 0 :
        // Isotropic
        CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet0<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0tet_gpu, F1tet_gpu, F2tet_gpu, F3tet_gpu/*, blabla*/);
        break;

    case 1 :
        // Anisotropic
        CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet1<<< grid1, threads1>>>(Lambda, Mu, nbElem, F0tet_gpu, F1tet_gpu, F2tet_gpu, F3tet_gpu);
        break;

    case 2 :
        // Viscoelastic
        CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet2<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1tet_gpu, Di2tet_gpu, Dv1tet_gpu, Dv2tet_gpu, F0tet_gpu, F1tet_gpu, F2tet_gpu, F3tet_gpu);
        break;

    case 3 :
        // Viscoelastic and anisotropic
        CudaTetrahedronTLEDForceField3f_calcForce_kernel_tet3<<< grid1, threads1>>>(Lambda, Mu, nbElem, Di1tet_gpu, Di2tet_gpu, Dv1tet_gpu, Dv2tet_gpu, F0tet_gpu, F1tet_gpu, F2tet_gpu, F3tet_gpu);
        break;
    }

    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    CudaTetrahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (float*)f/*, force*/);


//     mycudaMemcpyDeviceToHost(blablaCPU, blabla, 6*nbElem*sizeof(float));
//     for (int i = 0; i < nbElem; i++)
//     {
//         myprintf("el%d:    %f     %f     %f     %f     %f     %f\n", i, blablaCPU[6*i], blablaCPU[6*i+1], blablaCPU[6*i+2],
//                 blablaCPU[6*i+3], blablaCPU[6*i+4], blablaCPU[6*i+5]);
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

void CudaTetrahedronTLEDForceField3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    //CudaTetrahedronTLEDForceField3f_calcDForce_kernel<<< grid1, threads1>>>(nbElem, (const GPUElement*)elems, (GPUElementState*)state, (const float*)dx);
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    //CudaTetrahedronTLEDForceField3f_addForce_kernel<<< grid2, threads2, BSIZE*3*sizeof(float) >>>(nbVertex, nbElemPerVertex, (const GPUElement*)elems, (GPUElementState*)state, (const int*)velems, (float*)df, (const float*)dx);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
