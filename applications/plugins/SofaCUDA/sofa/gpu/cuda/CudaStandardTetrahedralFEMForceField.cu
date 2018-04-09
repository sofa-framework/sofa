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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include "cuda.h"
#include <cstdio>

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
void StandardTetrahedralFEMForceFieldCuda3f_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, float paramArray0, float paramArray1);
void StandardTetrahedralFEMForceFieldCuda3f_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, float param0, float param1);

#ifdef SOFA_GPU_CUDA_DOUBLE
void StandardTetrahedralFEMForceFieldCuda3d_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, double paramArray0, double paramArray1);
void StandardTetrahedralFEMForceFieldCuda3d_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, double param0, double param1);
#endif
}

#define TETRA_SIZE 158
#define TRC 1
#define Jindex 2
#define LAMBDA 3
#define TRCSQUARE 4
#define DEFTENSOR 6
#define RESTVOLUME 31
#define VOLSCALE 32
#define SHAPEVECTOR 33
#define FIBERDIRECTION 45
#define SPKTENSORGENERAL 60
#define DEFGRAD 66
#define MATB 75
#define STRAINENERGY 147
#define TETRAINDICES 148
#define TETRAEDGES 152

#define EDGE_SIZE 11
#define DFDX 0
#define V0 9
#define V1 10

#define TIMING false

//////////////////////
// GPU-side methods //
//////////////////////
template <class real>
__device__ real dot(real* A, real* B, int n)
{
    real sum = 0;
    for(int i=0; i<n; i++)
    {
        real product = __fmul_rn(A[i],B[i]);
        sum = __fadd_rn(sum,product);
    }
    return sum;
}

template <class real>
__device__ void cross(real* a, real* b, real* solution)
{
    real product0 = __fmul_rn(a[1],b[2]);
    real product1 = __fmul_rn(a[2],b[1]);
    solution[0] = __fadd_rn(product0, -product1);

    product0 = __fmul_rn(a[2],b[0]);
    product1 = __fmul_rn(a[0],b[2]);
    solution[1] = __fadd_rn(product0, -product1);

    product0 = __fmul_rn(a[0],b[1]);
    product1 = __fmul_rn(a[1],b[0]);
    solution[2] = __fadd_rn(product0, -product1);
}

template <class real>
__device__ real determinant3x3(real m[][3], int n)
{
	return m[0][0]*m[1][1]*m[2][2]
	    				 + m[1][0]*m[2][1]*m[0][2]
	    				 + m[2][0]*m[0][1]*m[1][2]
	    				 - m[0][0]*m[2][1]*m[1][2]
	    				 - m[1][0]*m[0][1]*m[2][2]
	    				 - m[2][0]*m[1][1]*m[0][2];
}

template <class real>
__device__ void invert(real from[][3], real dest[][3], int n)
{
    real det = determinant3x3(from, n);

    real product0 = __fmul_rn(from[1][1], from[2][2]);
    real product1 = __fmul_rn(from[2][1], from[1][2]);
    real sum = __fadd_rn(product0, -product1);
    dest[0][0]= __fdiv_rn(sum,det);
    
    product0 = __fmul_rn(from[1][2], from[2][0]);
    product1 = __fmul_rn(from[2][2], from[1][0]);
    sum = __fadd_rn(product0, -product1);
    dest[1][0]= __fdiv_rn(sum,det);
    dest[0][1] = dest[1][0];

    product0 = __fmul_rn(from[1][0], from[2][1]);
    product1 = __fmul_rn(from[2][0], from[1][1]);
    sum = __fadd_rn(product0, -product1);
    dest[2][0]= __fdiv_rn(sum,det);
    dest[0][2] = dest[2][0];

    product0 = __fmul_rn(from[2][2], from[0][0]);
    product1 = __fmul_rn(from[0][2], from[2][0]);
    sum = __fadd_rn(product0, -product1);
    dest[1][1]= __fdiv_rn(sum,det);

    product0 = __fmul_rn(from[2][0], from[0][1]);
    product1 = __fmul_rn(from[0][0], from[2][1]);
    sum = __fadd_rn(product0, -product1);
    dest[2][1]= __fdiv_rn(sum,det);
    dest[1][2] = dest[2][1];

    product0 = __fmul_rn(from[0][0], from[1][1]);
    product1 = __fmul_rn(from[1][0], from[0][1]);
    sum = __fadd_rn(product0, -product1);
    dest[2][2]= __fdiv_rn(sum,det);
}

template <class real>
__global__ void StandardTetrahedralFEMForceFieldCuda_addForce_kernel(int size, real* tetraInfo, real* contribution, const real* x, bool anisotropy, const real* anisoDirection, float paramArray0, float paramArray1)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size)
    //for(int i=0; i<size; i++)
    {
        //const Tetrahedron &ta= _topology->getTetrahedron(i);
        unsigned int ta[4];
        ta[0] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES];
        ta[1] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 1];
        ta[2] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 2];
        ta[3] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 3];

        //x0 = x[ta[0]]
        real x0[3];
        x0[0] = x[ta[0]*3];
        x0[1] = x[ta[0]*3+1];
        x0[2] = x[ta[0]*3+2];

        //dp[0] = x[ta[1]] - x0;
        real dp[3][3];
        dp[0][0] = x[ta[1]*3] - x0[0];
        dp[0][1] = x[ta[1]*3+1] - x0[1];
        dp[0][2] = x[ta[1]*3+2] - x0[2];

        //sv = tetInfo->shapeVector[1]
        real sv[3];
        sv[0] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + 3];
        sv[1] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + 4];
        sv[2] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + 5];

        real defGrad[3][3];
        for (int k=0;k<3;++k) {
            for (int l=0;l<3;++l) { 
                //		tetInfo->deformationGradient[k][l]=dp[0][k]*sv[l];
                defGrad[k][l] = dp[0][k]*sv[l];
            }
        }

        for (int j=1;j<3;++j) {
            //dp[j]=x[ta[j+1]]-x0;
            dp[j][0] = x[ta[j+1]*3] - x0[0];
            dp[j][1] = x[ta[j+1]*3+1] - x0[1];
            dp[j][2] = x[ta[j+1]*3+2] - x0[2];

            //sv=tetInfo->shapeVector[j+1];
            sv[0] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(j+1)*3)];
            sv[1] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(j+1)*3+1)];
            sv[2] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(j+1)*3+2)];
   
            for (int k=0;k<3;++k) {
                for (int l=0;l<3;++l) {
                    //tetInfo->deformationGradient[k][l]+=dp[j][k]*sv[l];
                    defGrad[k][l] = defGrad[k][l] + dp[j][k] * sv[l];
                }
            }
        }

        tetraInfo[i*TETRA_SIZE+DEFGRAD] = defGrad[0][0];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 1] = defGrad[0][1];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 2] = defGrad[0][2];

        tetraInfo[i*TETRA_SIZE+DEFGRAD + 3] = defGrad[1][0];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 4] = defGrad[1][1];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 5] = defGrad[1][2];

        tetraInfo[i*TETRA_SIZE+DEFGRAD + 6] = defGrad[2][0];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 7] = defGrad[2][1];
        tetraInfo[i*TETRA_SIZE+DEFGRAD + 8] = defGrad[2][2];

        //Compute the matrix strain displacement B 6*3
        for(int alpha=0; alpha<4; alpha++)
        {
            //Coord sva=tetInfo->shapeVector[alpha];
            real sva[3];
            sva[0] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(alpha)*3)];
            sva[1] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(alpha)*3)+1];
            sva[2] = tetraInfo[i*TETRA_SIZE+(SHAPEVECTOR+(alpha)*3)+2];



            //Matrix63 matBa;
            real matBa[6][3];

            matBa[0][0] = defGrad[0][0]*sva[0];
            matBa[0][1] = defGrad[1][0]*sva[0];
            matBa[0][2] = defGrad[2][0]*sva[0];


            matBa[2][0] = defGrad[0][1]*sva[1];
            matBa[2][1] = defGrad[1][1]*sva[1];
            matBa[2][2] = defGrad[2][1]*sva[1];

            matBa[5][0] = defGrad[0][2]*sva[2];
            matBa[5][1] = defGrad[1][2]*sva[2];
            matBa[5][2] = defGrad[2][2]*sva[2];

            
            matBa[1][0] = (defGrad[0][0]*sva[1]+defGrad[0][1]*sva[0]);
            matBa[1][1] = (defGrad[1][0]*sva[1]+defGrad[1][1]*sva[0]);
            matBa[1][2] = (defGrad[2][0]*sva[1]+defGrad[2][1]*sva[0]);

            matBa[3][0] = (defGrad[0][2]*sva[0]+defGrad[0][0]*sva[2]);
            matBa[3][1] = (defGrad[1][2]*sva[0]+defGrad[1][0]*sva[2]);
            matBa[3][2] = (defGrad[2][2]*sva[0]+defGrad[2][0]*sva[2]);

            matBa[4][0] = (defGrad[0][1]*sva[2]+defGrad[0][2]*sva[1]);
            matBa[4][1] = (defGrad[1][1]*sva[2]+defGrad[1][2]*sva[1]);
            matBa[4][2] = (defGrad[2][1]*sva[2]+defGrad[2][2]*sva[1]);

            //tetInfo->matB[alpha]=matBa;
            for(int j=0; j<6; j++)
            {
                for(int k=0; k<3; k++)
                {
                    tetraInfo[i*TETRA_SIZE+MATB+(alpha*18)+(j*3)+k] = matBa[j][k];
                }
            }
        }

        /// compute the right Cauchy-Green deformation matrix
        real defTensor[3][3];
        for (int k=0;k<3;++k) {
            for (int l=k;l<3;++l) {
                //tetInfo->deformationTensor(k,l)=(tetInfo->deformationGradient(0,k)*tetInfo->deformationGradient(0,l)+
                //	tetInfo->deformationGradient(1,k)*tetInfo->deformationGradient(1,l)+
                //	tetInfo->deformationGradient(2,k)*tetInfo->deformationGradient(2,l));
                defTensor[k][l] = (defGrad[0][k]*defGrad[0][l] + defGrad[1][k]*defGrad[1][l] + defGrad[2][k]*defGrad[2][l]);
                defTensor[l][k] = defTensor[k][l];
            }
        }

        int index = 0;
        for(int j=0; j<3; j++)
        {
            for(int k=0; k<j+1; k++)
            {
                tetraInfo[i*TETRA_SIZE+DEFTENSOR+index] = defTensor[j][k];
                index++;
            }
        }


        //in case of transversaly isotropy
        real fiberDirection[3];
        //if(globalParameters.anisotropyDirection.size()>0){
        if(anisotropy)
        {
            //	tetInfo->fiberDirection=globalParameters.anisotropyDirection[0];
            fiberDirection[0] = anisoDirection[0];
            fiberDirection[1] = anisoDirection[1];
            fiberDirection[2] = anisoDirection[2];

            tetraInfo[i*TETRA_SIZE+FIBERDIRECTION] = fiberDirection[0];
            tetraInfo[i*TETRA_SIZE+FIBERDIRECTION+1] = fiberDirection[1];
            tetraInfo[i*TETRA_SIZE+FIBERDIRECTION+2] = fiberDirection[2];

            //	Coord vectCa=tetInfo->deformationTensor*tetInfo->fiberDirection;
            real vectCa[3];
            for(int k=0; k<3; k++)
            {
                vectCa[k] = 0;
                for(int j=0; j<3; j++)
                {
                    vectCa[k] += defTensor[k][j] * fiberDirection[j];
                }
            }

            //	Real aDotCDota=dot(tetInfo->fiberDirection,vectCa);
            real aDotCDota = dot(fiberDirection, vectCa, 3);

            //	tetInfo->lambda=(Real)sqrt(aDotCDota);
            tetraInfo[i*TETRA_SIZE+LAMBDA] = sqrt(aDotCDota);
        }

        //Coord areaVec = cross( dp[1], dp[2] );
        real areaVec[3];
        cross(dp[1], dp[2], areaVec);

        //tetInfo->J = dot( areaVec, dp[0] ) * tetInfo->volScale;
        real volScale = tetraInfo[i*TETRA_SIZE + VOLSCALE];
        real J = dot(areaVec, dp[0], 3) * volScale;
        tetraInfo[i*TETRA_SIZE+Jindex] = J;

        //tetInfo->trC = (Real)( tetInfo->deformationTensor(0,0) + tetInfo->deformationTensor(1,1) + tetInfo->deformationTensor(2,2));        
        real trC = (defTensor[0][0] + defTensor[1][1] + defTensor[2][2]);
        tetraInfo[i*TETRA_SIZE+TRC] = trC;

        //tetInfo->SPKTensorGeneral.clear();
        for(int k=0; k<6; k++)
        {
            tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+k] = 0;
        }

        //MatrixSym SPK;
        //myMaterial->deriveSPKTensor(tetInfo,globalParameters,SPK); // calculate the SPK tensor of the chosen material
        //tetInfo->SPKTensorGeneral=SPK;


        StandardTetrahedralFEMForceFieldCuda_BoyceAndArruda_deriveSPKTensor(i, tetraInfo, paramArray0, paramArray1);

        real SPK[3][3];
        index=0;

        SPK[0][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 0];
        SPK[1][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 1];
        SPK[0][1] = SPK[1][0];
        SPK[1][1] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 2];
        SPK[2][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 3];
        SPK[0][2] = SPK[2][0];
        SPK[2][1] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 4];
        SPK[1][2] = SPK[2][1];
        SPK[2][2] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 5];

        real restVolume = tetraInfo[i*TETRA_SIZE+RESTVOLUME];

        for(int l=0;l<4;++l)
        {
            //f[ta[l]]-=tetInfo->matB[l].transposed()*SPK*tetInfo->restVolume;
              real result[3];
              real matBlT[3][6];
              for(int j=0; j<6; j++)
              {
                  for(int k=0; k<3; k++)
                  {
                      matBlT[k][j] = tetraInfo[i*TETRA_SIZE+MATB+(l*18)+(j*3)+k];
                  }
              }

              for(int j=0; j<3; j++)
              {
                  result[j] = matBlT[j][0] * SPK[0][0];
                  result[j] += matBlT[j][1] * SPK[1][0];
                  result[j] += matBlT[j][2] * SPK[1][1];
                  result[j] += matBlT[j][3] * SPK[2][0];
                  result[j] += matBlT[j][4] * SPK[2][1];
                  result[j] += matBlT[j][5] * SPK[2][2];
              }

			  result[0] = result[0] * restVolume;
			  result[1] = result[1] * restVolume;
			  result[2] = result[2] * restVolume;


			  contribution[i*12 + l*3 +0] = -(result[0]);
			  contribution[i*12 + l*3 +1] = -(result[1]);
			  contribution[i*12 + l*3 +2] = -(result[2]);

        }
    }   
}

template <class real>
__global__ void StandardTetrahedralFEMForceFieldCuda_PointsAccumulation_kernel(int nbPoints, int nbMaxTetraPerNode, const real* contribTetra, const int* neighbourhoodPoints, real* f)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < nbPoints)
    //for(int i=0; i<nbPoints; i++)
    {
        int indexNeighbourhood = i*nbMaxTetraPerNode;
        real acc[3];
        acc[0] = 0; acc[1] = 0; acc[2] = 0;

        for(int j=0; j < nbMaxTetraPerNode; j++)
        {
            int indexTetra = neighbourhoodPoints[indexNeighbourhood+j];
            if(indexTetra == -1)
                break;
            else
            {
                acc[0] += contribTetra[indexTetra*3];
                acc[1] += contribTetra[indexTetra*3+1];
                acc[2] += contribTetra[indexTetra*3+2];
            }
        }

        f[i*3+0] += acc[0];
        f[i*3+1] += acc[1];
        f[i*3+2] += acc[2];
    }
}

template <class real>
__global__ void StandardTetrahedralFEMForceFieldCuda_addDForce_PointsAccumulation_kernel(int nbEdges, int nbMaxTetraPerEdge, const real* contribDfDx, const int* neighbourhoodEdges, real* edgeInfo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < nbEdges)
    //for(int i=0; i<nbEdges; i++)
    {
        int indexNeighbourhood = i*nbMaxTetraPerEdge;
        real acc[3][3];
        for(int j=0; j<3; j++)
        {
            for(int k=0; k<3; k++)
            {
                acc[j][k] = 0;
            }
        }

        for(int j=0; j < nbMaxTetraPerEdge; j++)
        {
            int indexTetra = neighbourhoodEdges[indexNeighbourhood + j];
            if(indexTetra == -1)
                break;
            else
            {
                acc[0][0] += contribDfDx[indexTetra*9 + 0];
                acc[0][1] += contribDfDx[indexTetra*9 + 1];
                acc[0][2] += contribDfDx[indexTetra*9 + 2];
                acc[1][0] += contribDfDx[indexTetra*9 + 3];
                acc[1][1] += contribDfDx[indexTetra*9 + 4];
                acc[1][2] += contribDfDx[indexTetra*9 + 5];
                acc[2][0] += contribDfDx[indexTetra*9 + 6];
                acc[2][1] += contribDfDx[indexTetra*9 + 7];
                acc[2][2] += contribDfDx[indexTetra*9 + 8];
            }
        }

        edgeInfo[i*EDGE_SIZE + DFDX + 0] = acc[0][0];
        edgeInfo[i*EDGE_SIZE + DFDX + 1] = acc[0][1];
        edgeInfo[i*EDGE_SIZE + DFDX + 2] = acc[0][2];
        edgeInfo[i*EDGE_SIZE + DFDX + 3] = acc[1][0];
        edgeInfo[i*EDGE_SIZE + DFDX + 4] = acc[1][1];
        edgeInfo[i*EDGE_SIZE + DFDX + 5] = acc[1][2];
        edgeInfo[i*EDGE_SIZE + DFDX + 6] = acc[2][0];
        edgeInfo[i*EDGE_SIZE + DFDX + 7] = acc[2][1];
        edgeInfo[i*EDGE_SIZE + DFDX + 8] = acc[2][2];
    }
}

template <class real>
__global__ void StandardTetrahedralFEMForceFieldCuda_addDForce_kernel(int nbTetra, real* tetraInfo, real* edgeInfo, real param0, real param1, real* contribution)
{
    int maxEdgeIndex = 0;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < nbTetra)
//    for(int i=0; i<nbTetra; i++)
    {
        //const Tetrahedron &ta= _topology->getTetrahedron(i);
        unsigned int ta[4];
        ta[0] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 0];
        ta[1] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 1];
        ta[2] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 2];
        ta[3] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAINDICES + 3];

        //BaseMeshTopology::EdgesInTetrahedron te=_topology->getEdgesInTetrahedron(i);
        unsigned int te[6];
        te[0] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 0];
        te[1] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 1];
        te[2] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 2];
        te[3] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 3];
        te[4] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 4];
        te[5] = (unsigned int)tetraInfo[i*TETRA_SIZE + TETRAEDGES + 5];

        int localEdgesInTetrahedron[6][2]; 
        localEdgesInTetrahedron[0][0] = 0;
        localEdgesInTetrahedron[0][1] = 1;

        localEdgesInTetrahedron[1][0] = 0;
        localEdgesInTetrahedron[1][1] = 2;

        localEdgesInTetrahedron[2][0] = 0;
        localEdgesInTetrahedron[2][1] = 3;

        localEdgesInTetrahedron[3][0] = 1;
        localEdgesInTetrahedron[3][1] = 2;

        localEdgesInTetrahedron[4][0] = 1;
        localEdgesInTetrahedron[4][1] = 3;

        localEdgesInTetrahedron[5][0] = 2;
        localEdgesInTetrahedron[5][1] = 3;

        for(int j=0; j<6; j++)
        {
            if( te[j] > maxEdgeIndex)
            {
                maxEdgeIndex = te[j];
            }
            //Edge e=_topology->getLocalEdgesInTetrahedron(j);
			//k=e[0];
            int k = localEdgesInTetrahedron[j][0];
			//l=e[1];
            int l = localEdgesInTetrahedron[j][1];
			//if (edgeArray[te[j]][0]!=ta[k]) {
            if(edgeInfo[te[j]*EDGE_SIZE+V0] != ta[k])
            {
			    //  k=e[1];
                k=localEdgesInTetrahedron[j][1];
			    //  l=e[0];
                l=localEdgesInTetrahedron[j][0];
            }
            
			//einfo= &edgeInf[te[j]];
		    //Matrix3 &edgeDfDx = einfo->DfDx;
            real edgeDfDx[3][3];
            for(int n=0; n<3; n++)
            {
                for(int m=0; m<3; m++)
                    edgeDfDx[n][m] = edgeInfo[te[j]*EDGE_SIZE + DFDX + n*3 + m];
            }

			//Coord svl=tetInfo->shapeVector[l];
            real svl[3];
            svl[0] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + l*3 +0];
            svl[1] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + l*3 +1];
            svl[2] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + l*3 +2];
			//Coord svk=tetInfo->shapeVector[k];
            real svk[3];
            svk[0] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + k*3 +0];
            svk[1] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + k*3 +1];
            svk[2] = tetraInfo[i*TETRA_SIZE + SHAPEVECTOR + k*3 +2];

			//Matrix3  M, N;
			//Matrix6 outputTensor;
            real M[3][3];
            real N[3][3];
            real outputTensor[6][6];

			//N.clear();
            for(int m=0; m<3; m++)
            {
                for(int n=0; n<3; n++)
                    N[m][n] = 0;
            }

            StandardTetrahedralFEMForceFieldCuda_BoyceAndArruda_ElasticityTensor(i, tetraInfo, outputTensor, param0, param1);            

			//Matrix63 mBl=tetInfo->matB[l];
            real mBl[6][3];
            for(int n=0; n<6; n++)
            {
                for(int m=0; m<3; m++)
                {
                    mBl[n][m] = tetraInfo[i*TETRA_SIZE + MATB + l*18 + n*3 + m];
                }
            }
			//mBl[1][0]/=2;mBl[1][1]/=2;mBl[1][2]/=2;mBl[3][0]/=2;mBl[3][1]/=2;mBl[3][2]/=2;mBl[4][0]/=2;mBl[4][1]/=2;mBl[4][2]/=2;
			mBl[1][0]/=2;mBl[1][1]/=2;mBl[1][2]/=2;mBl[3][0]/=2;mBl[3][1]/=2;mBl[3][2]/=2;mBl[4][0]/=2;mBl[4][1]/=2;mBl[4][2]/=2;

            //N=(tetInfo->matB[k].transposed()*outputTensor*mBl);
            real matBkT[3][6];
            for(int n=0; n<6; n++)
            {
                for(int m=0; m<3; m++)
                {
                    matBkT[m][n] = tetraInfo[i*TETRA_SIZE + MATB + k*18 + n*3 + m];
                }
            }

            real temp[3][6];
            for(int n=0; n<3; n++)
            {
                for(int m=0; m<6; m++)
                {
                    temp[n][m] = matBkT[n][0]*outputTensor[0][m];
                    temp[n][m] += matBkT[n][1]*outputTensor[1][m];
                    temp[n][m] += matBkT[n][2]*outputTensor[2][m];
                    temp[n][m] += matBkT[n][3]*outputTensor[3][m];
                    temp[n][m] += matBkT[n][4]*outputTensor[4][m];
                    temp[n][m] += matBkT[n][5]*outputTensor[5][m];
                }
            }

            for(int n=0; n<3; n++)
            {
                for(int m=0; m<3; m++)
                {
                    N[n][m] = temp[n][0] * mBl[0][m];
                    N[n][m] += temp[n][1] * mBl[1][m];
                    N[n][m] += temp[n][2] * mBl[2][m];
                    N[n][m] += temp[n][3] * mBl[3][m];
                    N[n][m] += temp[n][4] * mBl[4][m];
                    N[n][m] += temp[n][5] * mBl[5][m];
                }
            }

            //Real productSD=0;
            real productSD = 0;
		    //Coord vectSD=tetInfo->SPKTensorGeneral*svk;
            real SPK[3][3];
            SPK[0][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+0];
            SPK[1][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+1];
            SPK[0][1] = SPK[1][0];
            SPK[1][1] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 2];
            SPK[2][0] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+ 3];
            SPK[0][2] = SPK[2][0];
            SPK[2][1] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+4];
            SPK[1][2] = SPK[2][1];
            SPK[2][2] = tetraInfo[i*TETRA_SIZE+SPKTENSORGENERAL+5];

            real vectSD[3];
            for(int n=0; n<3; n++)
            {
                vectSD[n] = SPK[n][0]*svk[0];
                vectSD[n] += SPK[n][1]*svk[1];
                vectSD[n] += SPK[n][2]*svk[2];
            }
			
            productSD=dot(vectSD,svl,3);
			//M[0][1]=M[0][2]=M[1][0]=M[1][2]=M[2][0]=M[2][1]=0;
			//M[0][0]=M[1][1]=M[2][2]=(Real)productSD;

           for(int n=0; n<3; n++)
           {
               for(int m=0; m<3; m++)
               {
                   M[n][m] = 0;
                   if(n==m)
                       M[n][m] = productSD;
               }
           }
    
		   //edgeDfDx += (M+N.transposed())*tetInfo->restVolume;
           real restVolume = tetraInfo[i*TETRA_SIZE+RESTVOLUME];
           real result[3][3];
           for(int n=0; n<3; n++)
           {
               for(int m=0; m<3; m++)
               {
                   result[n][m] = M[n][m] + N[m][n];
                   result[n][m] *= restVolume;
                   edgeDfDx[n][m] += result[n][m];
               }
		   }

		   //for(int n=0; n<3; n++)
		   //{
		   //    for(int m=0; m<3; m++)
		   //        contribution[(i*54)+(j*9)+(n*3)+m] = result[n][m];
		   //}
		   contribution[(i*54)+(j*9)+0] = result[0][0];
		   contribution[(i*54)+(j*9)+1] = result[0][1];
		   contribution[(i*54)+(j*9)+2] = result[0][2];
		   contribution[(i*54)+(j*9)+3] = result[1][0];
		   contribution[(i*54)+(j*9)+4] = result[1][1];
		   contribution[(i*54)+(j*9)+5] = result[1][2];
		   contribution[(i*54)+(j*9)+6] = result[2][0];
		   contribution[(i*54)+(j*9)+7] = result[2][1];
		   contribution[(i*54)+(j*9)+8] = result[2][2];

		}
	}
}

template <class real>
__device__ void StandardTetrahedralFEMForceFieldCuda_BoyceAndArruda_deriveSPKTensor(int tetraIndex, real* tetraInfo, float paramArray0, float paramArray1)
{
    //MatrixSym C = sinfo->deformationTensor
    real C[3][3];
    int index=0;
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<j+1; k++)
        {
            C[j][k] = tetraInfo[tetraIndex*TETRA_SIZE+DEFTENSOR+index];
            C[k][j] = C[j][k];
            index++;
        }
    }
    //invertMatrix(inversematrix, C);
    real inversematrix[3][3];
    invert(C, inversematrix, 3);

    //Real I1 = sinfo->trC
    real I1 = tetraInfo[tetraIndex*TETRA_SIZE+TRC];
    //Real J = sinfo->J;
    real J = tetraInfo[tetraIndex*TETRA_SIZE+Jindex];
    //Real mu=param.parameterArray[0]
    real mu = paramArray0;
    //Real k0=param.parameterArray[1];
    real k0 = paramArray1;

    //MatrixSym ID
    real ID[3][3];
    //ID.identity()
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            if(j==k)
                ID[j][k] = 1;
            else
                ID[j][k] = 0;
        }
    }

    //SPKTensorGeneral=((inversematrix*(Real)(-1.0/3.0)*I1+ID)*(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))+(inversematrix*(Real)(-2.0/3.0)*I1*I1+ID*(Real)2.0*I1)*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))+(ID*(Real)3.0*I1*I1-inversematrix*I1*I1*I1)*(Real)(11.0/(1050.0*8*8))*pow(sinfo->J,(Real)(-2.0))
    ////	+(inversematrix*(Real)(-4.0/3.0)*pow(I1,(Real)4.0)+ID*(Real)4.0*pow(I1,(Real)3.0))*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))+(inversematrix*(Real)(-5.0/3.0)*pow(I1,(Real)5.0)+ID*(Real)5.0*pow(I1,(Real)4.0))*(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0)))*2.0*mu
    ////	+inversematrix*k0*log(sinfo->J);

    //result = inversematrix * (-1.0/3.0) * I1
    // + ID
    // * (1/2) * J^(-2/3) 
    real result[3][3];
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            result[j][k] = inversematrix[j][k] * (-1.0/3.0) * I1;
            result[j][k] = result[j][k] + ID[j][k];
            result[j][k] = result[j][k] * (1.0/2.0) * pow(J, (real)(-2.0/3.0));
        }
    }

    // result += ((inversematrix * (-2/3) * I1 * I1) + (ID * 2 * I1)) * (1/160) * J^(-4/3)
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            real temp = inversematrix[j][k] * (-2.0/3.0) * I1 * I1;
            real temp2 = ID[j][k] * 2.0 * I1;
            temp = temp + temp2;
            temp = temp * (1.0/160.0) * pow(J, (real)(-4.0/3.0));

            result[j][k] += temp;
        }
    }

    // result += ((ID * 3 * I1 * I1) - (inversematrix * I1^3)) * (11/(1050*8*8)) * J^-2
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            real temp = ID[j][k] * 3.0 * I1 * I1;
            real temp2 = inversematrix[j][k] * pow(I1,3);
            temp = temp - temp2;
            temp = temp * (11.0/(1050.0*8.0*8.0)) * pow(J, -2);
            result[j][k] += temp;
        }
    }

    // result += ((inversematrix * (-4/3) * I1^4) + (ID * 4 * I1^3)) * (19 / (7000 * 8 * 8 * 8)) * J^(-8/3)
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            real temp = inversematrix[j][k] * (real)(-4.0/3.0) * pow(I1,(real)4.0);
            real temp2 = ID[j][k] * (real)4.0 * pow(I1,(real)3.0);
            temp = temp+temp2;
            temp = temp * (real)(19.0 / (7000.0*8.0*8.0*8.0) ) * pow(J, (real)(-8.0/3.0));
            result[j][k] += temp;
        }

    }

    // result += ((inversematrix[j][k] * (-5/3) * I1^5) + (ID * 5 * I1^4)) * (519 / (673750*8*8*8*8)) * J^(-10/3)
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            real temp = inversematrix[j][k] * (-5.0/3.0) * pow(I1, 5);
            real temp2 = ID[j][k] * 5.0 * pow(I1, 4);
            temp = temp + temp2;
            temp = temp * (519.0 / (673750.0* pow((real) 8,4))) * pow(J, (real)(-10.0/3.0));
            result[j][k] += temp;
        }
    }

    // result = (result * 2 * mu) + (inversematrix * k0 * log(J))
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            //Here the error becomes greater than 1e-6
            result[j][k] = result[j][k] * (real)2.0 * mu;
            real temp = inversematrix[j][k] * k0 * log(J);
            result[j][k] += temp;    
        }
    }

    //SPKTensorGeneral = result;
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL] = result[0][0];
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL + 1] = result[1][0];
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL + 2] = result[1][1];
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL + 3] = result[2][0];
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL + 4] = result[2][1];
    tetraInfo[tetraIndex*TETRA_SIZE+SPKTENSORGENERAL + 5] = result[2][2];
}

template <class real>
__device__ void StandardTetrahedralFEMForceFieldCuda_BoyceAndArruda_ElasticityTensor(int tetraIndex, real* tetraInfo, real outputTensor[][6], real param0, real param1)
{
    //MatrixSym ID;
    //ID.identity();
    real ID[3][3];
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            if(j==k)
                ID[j][k] = 1;
            else
                ID[j][k] = 0;
        }
    }

    //MatrixSym _C;
    real _C[3][3];

    //invertMatrix(_C,sinfo->deformationTensor);
    real defTensor[3][3];
    int index=0;
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<j+1; k++)
        {
            defTensor[j][k] = tetraInfo[tetraIndex*TETRA_SIZE+DEFTENSOR+index];
            defTensor[k][j] = defTensor[j][k];
            index++;
        }
    }

    invert(defTensor, _C, 3);



    //MatrixSym CC;
    real CC[3][3];
    //CC=_C;
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<3; k++)
        {
            CC[j][k] = _C[j][k];
        }
    }    


    //CC[1]+=_C[1];CC[3]+=_C[3];CC[4]+=_C[4];
    CC[1][0] += _C[1][0];
    CC[0][1] = CC[1][0];
    CC[2][0] += _C[2][0];
    CC[0][2] = CC[2][0];
    CC[2][1] += _C[2][1];
    CC[1][2] = CC[2][1];

    real _C_reduced[6];
    _C_reduced[0] = _C[0][0];
    _C_reduced[1] = _C[1][0];
    _C_reduced[2] = _C[1][1];
    _C_reduced[3] = _C[2][0];
    _C_reduced[4] = _C[2][1];
    _C_reduced[5] = _C[2][2];

    //Matrix6 C_H_C;
    real C_H_C[6][6];
    //C_H_C[0][0]=_C[0]*_C[0]; 
    C_H_C[0][0] = _C_reduced[0]*_C_reduced[0];
    //C_H_C[1][1]=_C[1]*_C[1]+_C[0]*_C[2]; 
    C_H_C[1][1] = _C_reduced[1]*_C_reduced[1] + _C_reduced[0]*_C_reduced[2];
    //C_H_C[2][2]=_C[2]*_C[2];
    C_H_C[2][2] = _C_reduced[2]*_C_reduced[2];
    //C_H_C[3][3]=_C[3]*_C[3]+_C[0]*_C[5];
    C_H_C[3][3] = _C_reduced[3]*_C_reduced[3] + _C_reduced[0]*_C_reduced[5];
    //C_H_C[4][4]=_C[4]*_C[4]+_C[2]*_C[5];
    C_H_C[4][4] = _C_reduced[4]*_C_reduced[4] + _C_reduced[2]*_C_reduced[5];
    //C_H_C[5][5]=_C[5]*_C[5];
    C_H_C[5][5] = _C_reduced[5]*_C_reduced[5];
    //C_H_C[1][0]=_C[0]*_C[1];
    C_H_C[1][0] = _C[0][0]*_C[1][0];
    //C_H_C[0][1]=2*C_H_C[1][0]; 
    C_H_C[0][1] = 2 * C_H_C[1][0];
    //C_H_C[2][0]=C_H_C[0][2]=_C[1]*_C[1];
    C_H_C[2][0] = _C[1][0] * _C[1][0];
    C_H_C[0][2] = C_H_C[2][0];
    //C_H_C[5][0]=C_H_C[0][5]=_C[3]*_C[3];
    C_H_C[5][0] = _C[2][0] * _C[2][0];
    C_H_C[0][5] = C_H_C[5][0];
    //C_H_C[3][0]=_C[0]*_C[3];
    C_H_C[3][0] = _C[0][0]*_C[2][0];
    //C_H_C[0][3]=2*C_H_C[3][0];
    C_H_C[0][3] = 2*C_H_C[3][0];
    //C_H_C[4][0]=_C[1]*_C[3];
    C_H_C[4][0] = _C[1][0] * _C[2][0];
    //C_H_C[0][4]=2*C_H_C[4][0];
    C_H_C[0][4] = 2*C_H_C[4][0];
    //C_H_C[1][2]=_C[2]*_C[1];
    C_H_C[1][2] = _C[1][1] * _C[1][0];
    //C_H_C[2][1]=2*C_H_C[1][2];
    C_H_C[2][1] = 2*C_H_C[1][2];
    //C_H_C[1][5]=_C[3]*_C[4];
    C_H_C[1][5] = _C[2][0] * _C[2][1];
    //C_H_C[5][1]=2*C_H_C[1][5];
    C_H_C[5][1] = 2*C_H_C[1][5];
    //C_H_C[3][1]=C_H_C[1][3]=_C[0]*_C[4]+_C[1]*_C[3];
    C_H_C[3][1] = _C[0][0]*_C[2][1] + _C[1][0]*_C[2][0];
    C_H_C[1][3] = C_H_C[3][1];
    //C_H_C[1][4]=C_H_C[4][1]=_C[1]*_C[4]+_C[2]*_C[3];
    C_H_C[1][4] = _C[1][0]*_C[2][1] + _C[1][1]*_C[2][0];
    C_H_C[4][1] = C_H_C[1][4];
    //C_H_C[3][2]=_C[4]*_C[1];
    C_H_C[3][2] = _C[2][1] * _C[1][0];
    //C_H_C[2][3]=2*C_H_C[3][2];
    C_H_C[2][3] = 2*C_H_C[3][2];
    //C_H_C[4][2]=_C[4]*_C[2];
    C_H_C[4][2] = _C[2][1]*_C[1][1];
    //C_H_C[2][4]=2*C_H_C[4][2];
    C_H_C[2][4] = 2*C_H_C[4][2];
    //C_H_C[2][5]=C_H_C[5][2]=_C[4]*_C[4];
    C_H_C[2][5] = _C[2][1] * _C[2][1];
    C_H_C[5][2] = C_H_C[2][5];
    //C_H_C[3][5]=_C[3]*_C[5];
    C_H_C[3][5] = _C[2][0] * _C[2][2];
    //C_H_C[5][3]=2*C_H_C[3][5];
    C_H_C[5][3] = 2*C_H_C[3][5];
    //C_H_C[4][3]=C_H_C[3][4]=_C[3]*_C[4]+_C[5]*_C[1];
    C_H_C[4][3] = _C[2][0]*_C[2][1] + _C[2][2]*_C[1][0];
    C_H_C[3][4] = C_H_C[4][3];
    //C_H_C[4][5]=_C[4]*_C[5];
    C_H_C[4][5] = _C[2][1]*_C[2][2];
    //C_H_C[5][4]=2*C_H_C[4][5];
    C_H_C[5][4] = 2*C_H_C[4][5];


    real ID_reduced[6];
    ID_reduced[0] = ID[0][0];
    ID_reduced[1] = ID[1][0];
    ID_reduced[2] = ID[1][1];
    ID_reduced[3] = ID[2][0];
    ID_reduced[4] = ID[2][1];
    ID_reduced[5] = ID[2][2];


    //Matrix6 trC_HC_;
    real trC_HC_[6][6];
    //trC_HC_[0]=_C[0]*CC;
    //trC_HC_[1]=_C[1]*CC;
    //trC_HC_[2]=_C[2]*CC;
    //trC_HC_[3]=_C[3]*CC;
    //trC_HC_[4]=_C[4]*CC;
    //trC_HC_[5]=_C[5]*CC; 
    //trC_HC_[0][0] = _C_reduced[0]*CC[0][0];
    for(int k=0; k<6; k++)
    {
        trC_HC_[k][0] = _C_reduced[k]*CC[0][0];
        trC_HC_[k][1] = _C_reduced[k]*CC[1][0];
        trC_HC_[k][2] = _C_reduced[k]*CC[1][1];
        trC_HC_[k][3] = _C_reduced[k]*CC[2][0];
        trC_HC_[k][4] = _C_reduced[k]*CC[2][1];
        trC_HC_[k][5] = _C_reduced[k]*CC[2][2];
    }

    //Matrix6 trID_HC_;
    real trID_HC_[6][6];
    //trID_HC_[0]=ID[0]*CC;
    //trID_HC_[1]=ID[1]*CC;
    //trID_HC_[2]=ID[2]*CC;
    //trID_HC_[3]=ID[3]*CC;
    //trID_HC_[4]=ID[4]*CC;
    //trID_HC_[5]=ID[5]*CC;
    for(int k=0; k<6; k++)
    {
        trID_HC_[k][0] = ID_reduced[k]*CC[0][0];
        trID_HC_[k][1] = ID_reduced[k]*CC[1][0];
        trID_HC_[k][2] = ID_reduced[k]*CC[1][1];
        trID_HC_[k][3] = ID_reduced[k]*CC[2][0];
        trID_HC_[k][4] = ID_reduced[k]*CC[2][1];
        trID_HC_[k][5] = ID_reduced[k]*CC[2][2];
    }
    //Matrix6 trC_HID;
    real trC_HID[6][6];
    //trC_HID[0]=_C[0]*ID;
    //trC_HID[1]=_C[1]*ID;
    //trC_HID[2]=_C[2]*ID;
    //trC_HID[3]=_C[3]*ID;
    //trC_HID[4]=_C[4]*ID;
    //trC_HID[5]=_C[5]*ID;
    for(int k=0; k<6; k++)
    {
        trC_HID[k][0] = _C_reduced[k]*ID[0][0];
        trC_HID[k][1] = _C_reduced[k]*ID[1][0];
        trC_HID[k][2] = _C_reduced[k]*ID[1][1];
        trC_HID[k][3] = _C_reduced[k]*ID[2][0];
        trC_HID[k][4] = _C_reduced[k]*ID[2][1];
        trC_HID[k][5] = _C_reduced[k]*ID[2][2];
    }
    //Matrix6 trIDHID;
    real trIDHID[6][6];
    //trIDHID[0]=ID[0]*ID;
    //trIDHID[1]=ID[1]*ID;
    //trIDHID[2]=ID[2]*ID;
    //trIDHID[3]=ID[3]*ID;
    //trIDHID[4]=ID[4]*ID;
    //trIDHID[5]=ID[5]*ID;
    for(int k=0; k<6; k++)
    {
        trIDHID[0][k] = ID_reduced[0]*ID_reduced[k];
        trIDHID[1][k] = ID_reduced[1]*ID_reduced[k];
        trIDHID[2][k] = ID_reduced[2]*ID_reduced[k];
        trIDHID[3][k] = ID_reduced[3]*ID_reduced[k];
        trIDHID[4][k] = ID_reduced[4]*ID_reduced[k];
        trIDHID[5][k] = ID_reduced[5]*ID_reduced[k];
    }
    //Real I1=sinfo->trC;
    real I1 = tetraInfo[tetraIndex*TETRA_SIZE+TRC];
    real J = tetraInfo[tetraIndex*TETRA_SIZE+Jindex];

    //Real mu=param.parameterArray[0];
    real mu = param0;
    //Real k0=param.parameterArray[1];
    real k0 = param1;

    //outputTensor=((((trC_HC_*(Real)(1.0/3.0)*I1-trID_HC_)*(Real)(1.0/3.0)-trC_HID*(Real)(1.0/3.0)+C_H_C*(Real)(1.0/3.0)*I1)*(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))
    //	+((trC_HC_*(Real)(2.0/3.0)*I1*I1-trID_HC_*(Real)2.0*I1)*(Real)(2.0/3.0)-trC_HID*(Real)(4.0/3.0)*I1+C_H_C*(Real)(2.0/3.0)*I1*I1+trIDHID*(Real)2.0)*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))
    //	+(trC_HC_*I1*I1*I1-trID_HC_*(Real)3.0*I1*I1-trC_HID*(Real)3.0*I1*I1+C_H_C*I1*I1*I1+trIDHID*(Real)6.0*I1)*(Real)(11.0/(1050.0*8.0*8.0))*pow(sinfo->J,(Real)-2.0)
    //	+((trC_HC_*(Real)(4.0/3.0)*pow(I1,(Real)4.0)-trID_HC_*(Real)4.0*pow(I1,(Real)3.0))*(Real)(4.0/3.0)-trC_HID*(Real)(16.0/3.0)*pow(I1,(Real)3.0)+C_H_C*(Real)(4.0/3.0)*pow(I1,(Real)4.0)+trIDHID*(Real)12.0*I1*I1)*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))
    //	+((trC_HC_*(Real)(5.0/3.0)*pow(I1,(Real)5.0)-trID_HC_*(Real)5*pow(I1,(Real)4.0))*(Real)(5.0/3.0)-trC_HID*(Real)(25.0/3.0)*pow(I1,(Real)4.0)+C_H_C*(Real)(5.0/3.0)*pow(I1,(Real)5.0)+trIDHID*(Real)20.0*pow(I1,(Real)3.0))*(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0)))*2.0*mu
    //	+trC_HC_*(Real)k0/(Real)2.0-C_H_C*k0*log(sinfo->J))*2.0;

    // (trC_HC_*(Real)(1.0/3.0)*I1 - trID_HC_)
    //  *(Real)(1.0/3.0)
    //  - trC_HID*(Real)(1.0/3.0)
    //  + C_H_C*(Real)(1.0/3.0)*I1
    //  )
    //  *(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))

    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            outputTensor[j][k] = trC_HC_[j][k] *(1.0/3.0) * I1;
            outputTensor[j][k] = outputTensor[j][k] - trID_HC_[j][k];
            outputTensor[j][k] = outputTensor[j][k] * (1.0/3.0);
            real temp = trC_HID[j][k] * (1.0/3.0);
            outputTensor[j][k] = outputTensor[j][k] - temp;
            temp = C_H_C[j][k] * (1.0/3.0) * I1;
            outputTensor[j][k] = outputTensor[j][k] + temp;
            outputTensor[j][k] = outputTensor[j][k] * (1.0/2.0) * pow(J, (real)(-2.0/3.0));
        }
    }

    //   ((
    //      trC_HC_*(Real)(2.0/3.0)*I1*I1
    //      - trID_HC_*(Real)2.0*I1
    //   )*(Real)(2.0/3.0)
    //   - trC_HID*(Real)(4.0/3.0)*I1
    //   + C_H_C*(Real)(2.0/3.0)*I1*I1
    //   + trIDHID*(Real)2.0
    // )*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))
    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            real temp = trC_HC_[j][k] * (2.0/3.0) * I1 * I1;
            real temp2 = trID_HC_[j][k] * 2.0 * I1;
            temp = temp - temp2;
            temp = temp * (2.0/3.0);
            temp2 = trC_HID[j][k] * (4.0/3.0) * I1;
            temp = temp - temp2;
            temp2 = C_H_C[j][k] * (2.0/3.0) * I1 * I1;
            temp = temp + temp2;
            temp2 = trIDHID[j][k] * 2.0;
            temp = temp + temp2;
            temp = temp * (1.0/160.0) * pow(J, (real)(-4.0/3.0));
            outputTensor[j][k] = outputTensor[j][k] + temp;
        }
    }

    //(
    //    trC_HC_*I1*I1*I1
    //    -
    //    trID_HC_*(Real)3.0*I1*I1
    //    -
    //    trC_HID*(Real)3.0*I1*I1
    //    +
    //    C_H_C*I1*I1*I1
    //    +
    //    trIDHID*(Real)6.0*I1
    // )
    // *(Real)(11.0/(1050.0*8.0*8.0))*pow(sinfo->J,(Real)-2.0)
    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            real temp = trC_HC_[j][k] * I1 * I1 * I1;
            real temp2 = trID_HC_[j][k] * 3.0 * I1 * I1;
            temp = temp - temp2;
            temp2 = trC_HID[j][k] * 3.0 * I1 * I1;
            temp = temp - temp2;
            temp2 = C_H_C[j][k] * I1 * I1 * I1;
            temp = temp + temp2;
            temp2 = trIDHID[j][k] * 6.0 * I1;
            temp = temp + temp2;
            temp = temp * (11.0/(1050.0*8.0*8.0)) * pow(J, (real)-2.0);
            outputTensor[j][k] = outputTensor[j][k] + temp;
        }
    }
    //  ((
    //      trC_HC_*(Real)(4.0/3.0)*pow(I1,(Real)4.0)
    //      - trID_HC_*(Real)4.0*pow(I1,(Real)3.0)
    //   )
    //   *(Real)(4.0/3.0)
    //   - trC_HID*(Real)(16.0/3.0)*pow(I1,(Real)3.0)
    //   + C_H_C*(Real)(4.0/3.0)*pow(I1,(Real)4.0)
    //   + trIDHID*(Real)12.0*I1*I1
    // )*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))

    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            real temp = trC_HC_[j][k] * (4.0/3.0) * pow(I1, (real)4.0);
            real temp2 = trID_HC_[j][k] * 4.0 * pow(I1, (real)3.0);
            temp = temp - temp2;
            temp = temp * (4.0/3.0);
            temp2 = trC_HID[j][k] * (16.0/3.0);
            temp2 = temp2 * pow(I1, (real)3.0);
            temp = temp - temp2;
            temp2 = C_H_C[j][k] * (4.0/3.0) * pow(I1,(real)4.0);
            temp = temp + temp2;
            temp2 = trIDHID[j][k] * 12.0 * I1 * I1;
            temp = temp + temp2;
            temp = temp * (19.0/(7000.0*8.0*8.0*8.0)) * pow(J,(real)(-8.0/3.0));
            outputTensor[j][k] = outputTensor[j][k] + temp;
        }
    }
    //  ((
    //     trC_HC_*(Real)(5.0/3.0)*pow(I1,(Real)5.0)
    //     - trID_HC_*(Real)5*pow(I1,(Real)4.0)
    //  )*(Real)(5.0/3.0)
    //  - trC_HID*(Real)(25.0/3.0)*pow(I1,(Real)4.0)
    //  + C_H_C*(Real)(5.0/3.0)*pow(I1,(Real)5.0)
    //  + trIDHID*(Real)20.0*pow(I1,(Real)3.0)
    //  )* (Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0))
    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            real temp = trC_HC_[j][k] * (5.0/3.0) * pow(I1,(real)5.0);
            real temp2 = trID_HC_[j][k] * 5.0 * pow(I1,(real)4.0);
            temp = temp - temp2;
            temp = temp * (5.0/3.0);
            temp2 = trC_HID[j][k] * (25.0/3.0) * pow(I1,(real)4.0);
            temp = temp - temp2;
            temp2 = C_H_C[j][k] * (5.0/3.0) * pow(I1,(real)5.0);
            temp = temp + temp2;
            temp2 = trIDHID[j][k] * 20.0 * pow(I1,(real)3.0);
            temp = temp + temp2;
            temp = temp * (519.0/(673750.0*8.0*8.0*8.0*8.0)) * pow(J,(real)(-10.0/3.0));
            outputTensor[j][k] = outputTensor[j][k] + temp;
        }
    }
    //*
    //2.0*mu
    //+
    // trC_HC_*(Real)k0/(Real)2.0
    // -
    // C_H_C*k0*log(sinfo->J)
    // )*2.0;
    for(int j=0; j<6; j++)
    {
        for(int k=0; k<6; k++)
        {
            outputTensor[j][k] = outputTensor[j][k] * 2.0 * mu;
            outputTensor[j][k] = outputTensor[j][k] + trC_HC_[j][k] * k0/2.0;
            outputTensor[j][k] = outputTensor[j][k] - C_H_C[j][k] * k0 * log(J);
            outputTensor[j][k] = outputTensor[j][k] * 2.0;
        }
    }
}

template <class real>
__device__ void createMatrix(real** matrix, int n, int m)
{
    matrix = (real**)malloc(n*sizeof(real*));
    for(int i=0; i<n; i++)
        matrix[i] = (real*)malloc(m*sizeof(real));
}

template <class real>
__device__ void deleteMatrix(real** matrix, int n)
{
    for(int i=0; i<n; i++)
        free(matrix[i]);
    free(matrix);
}

//////////////////////
// CPU-side methods //
//////////////////////

void StandardTetrahedralFEMForceFieldCuda3f_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, float paramArray0, float paramArray1)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTetra+BSIZE-1)/BSIZE,1);
    
    StandardTetrahedralFEMForceFieldCuda_addForce_kernel<float><<<grid, threads>>>(nbTetra, (float*) tetraInfo, (float*)contribTetra, (const float*) x, anisotropy, (const float*) anisoDirection, paramArray0, paramArray1);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_addForce_kernel<float>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints+BSIZE-1)/BSIZE,1);

    StandardTetrahedralFEMForceFieldCuda_PointsAccumulation_kernel<float><<<grid2, threads>>>(nbPoints, nbMaxTetraPerNode, (const float*) contribTetra, (const int*) neighbourhoodPoints, (float*) f);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_GPUGEMS_PointsAccumulation_kernel<float>");

    if(TIMING)
        cudaDeviceSynchronize();
}

void StandardTetrahedralFEMForceFieldCuda3f_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, float param0, float param1)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTetra+BSIZE-1)/BSIZE,1);
    
    StandardTetrahedralFEMForceFieldCuda_addDForce_kernel<float><<<grid, threads>>>(nbTetra, (float*) tetraInfo, (float*)edgeInfo, param0, param1, (float*)contribDfDx);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_addDForce_kernel<float>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbEdges+BSIZE-1)/BSIZE,1);

    StandardTetrahedralFEMForceFieldCuda_addDForce_PointsAccumulation_kernel<float><<<grid2, threads2>>>(nbEdges, nbMaxTetraPerEdge, (const float*) contribDfDx, (const int*) neighbourhoodEdges, (float*) edgeInfo);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_GPUGEMS_addDForce_PointsAccumulation_kernel<float>");

    if(TIMING)
        cudaDeviceSynchronize();
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void StandardTetrahedralFEMForceFieldCuda3d_addForce(int nbTetra, int nbPoints, int nbMaxTetraPerNode, const void* neighbourhoodPoints, void* contribTetra, void* tetraInfo, void* f, const void* x, bool anisotropy, const void* anisoDirection, double paramArray0, double paramArray1)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTetra+BSIZE-1)/BSIZE,1);
    
    StandardTetrahedralFEMForceFieldCuda_addForce_kernel<double><<<grid, threads>>>(nbTetra, (double*) tetraInfo, (double*)contribTetra, (const double*) x, anisotropy, (const double*) anisoDirection, paramArray0, paramArray1);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_addForce_kernel<double>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints+BSIZE-1)/BSIZE,1);

    StandardTetrahedralFEMForceFieldCuda_PointsAccumulation_kernel<double><<<grid2, threads2>>>(nbPoints, nbMaxTetraPerNode, (const double*) contribTetra, (const int*) neighbourhoodPoints, (double*) f);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_PointsAccumulation_kernel<double>");

    if(TIMING)
        cudaDeviceSynchronize();
}

void StandardTetrahedralFEMForceFieldCuda3d_addDForce(int nbTetra, int nbEdges, int nbMaxTetraPerEdge, void* tetraInfo, void* edgeInfo, void* contribDfDx, const void* neighbourhoodEdges, double param0, double param1)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbTetra+BSIZE-1)/BSIZE,1);
    
    StandardTetrahedralFEMForceFieldCuda_addDForce_kernel<double><<<grid, threads>>>(nbTetra, (double*) tetraInfo, (double*)edgeInfo, param0, param1, (double*)contribDfDx);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_addDForce_kernel<double>");

    cudaDeviceSynchronize();

    dim3 threads2(BSIZE);
    dim3 grid2((nbEdges+BSIZE-1)/BSIZE,1);

    StandardTetrahedralFEMForceFieldCuda_addDForce_PointsAccumulation_kernel<double><<<grid2, threads2>>>(nbEdges, nbMaxTetraPerEdge, (const double*) contribDfDx, (const int*) neighbourhoodEdges, (double*) edgeInfo);
    mycudaDebugError("StandardTetrahedralFEMForceFieldCuda_addDForce_PointsAccumulation_kernel<double>");

    if(TIMING)
        cudaDeviceSynchronize();
}
#endif


#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
