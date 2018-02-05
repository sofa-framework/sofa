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
#include "cuda.h"
#include <cstdio>
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>

#define EDGE_SIZE 11
#define V0 9
#define V1 10
#define DFDX 0

#define TIMING true

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
void TetrahedralTensorMassForceFieldCuda3f_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo );
void TetrahedralTensorMassForceFieldCuda3f_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx, const void* edgeInfo, float kFactor );
#ifdef SOFA_GPU_CUDA_DOUBLE
void TetrahedralTensorMassForceFieldCuda3d_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo );
void TetrahedralTensorMassForceFieldCuda3d_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx, const void* edgeInfo, double kFactor );
#endif
}

//////////////////////
// GPU-side methods //
//////////////////////

template <class real>
__global__ void TetrahedralTensorMassForceFieldCuda_addForce_EdgeContribution_kernel(int size, real* contribEdge, const real* x, const real* initialPoints, const real* edgeInfo)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < size)
    {
        //v0=_topology->getEdge(i)[0];
        int v0 = (int) edgeInfo[i*EDGE_SIZE+V0];
        //v1=_topology->getEdge(i)[1];
        int v1 = (int) edgeInfo[i*EDGE_SIZE+V1];

        real dp0[3];
        real dp1[3];
        real dp[3];

        //dp0=x[v0]-_initialPoints[v0];
        dp0[0] = x[v0*3] - initialPoints[v0*3];
        dp0[1] = x[v0*3+1] - initialPoints[v0*3+1];
        dp0[2] = x[v0*3+2] - initialPoints[v0*3+2];

        //dp1=x[v1]-_initialPoints[v1];
        dp1[0] = x[v1*3] - initialPoints[v1*3];
        dp1[1] = x[v1*3+1] - initialPoints[v1*3+1];
        dp1[2] = x[v1*3+2] - initialPoints[v1*3+2];

        //dp = dp1-dp0;
        dp[0] = dp1[0] - dp0[0];
        dp[1] = dp1[1] - dp0[1];
        dp[2] = dp1[2] - dp0[2];

        const real* matrix = &edgeInfo[i*EDGE_SIZE+DFDX];

        //contribEdge[i][0]= -(einfo->DfDx.transposeMultiply(dp));
        contribEdge[i*6] = - (matrix[0]*dp[0] + matrix[3]*dp[1] + matrix[6]*dp[2]) ;
        contribEdge[i*6+1] = - (matrix[1]*dp[0] + matrix[4]*dp[1] + matrix[7]*dp[2]) ;
        contribEdge[i*6+2] = - (matrix[2]*dp[0] + matrix[5]*dp[1] + matrix[8]*dp[2]) ;

        //contribEdge[i][1]=einfo->DfDx*dp;
        contribEdge[i*6+3] = (matrix[0]*dp[0] + matrix[1]*dp[1] + matrix[2]*dp[2]);
        contribEdge[i*6+4] = (matrix[3]*dp[0] + matrix[4]*dp[1] + matrix[5]*dp[2]);
        contribEdge[i*6+5] = (matrix[6]*dp[0] + matrix[7]*dp[1] + matrix[8]*dp[2]);
		
   }

}

template <class real>
__global__ void TetrahedralTensorMassForceFieldCuda_addDForce_EdgeContribution_kernel(int size, real* contribEdge, const real* dx, const real* edgeInfo, real kFactor)
{
    /*	unsigned int v0,v1;
	EdgeRestInformation *einfo;
	Deriv force;
	Coord dp0,dp1,dp;

	for(int i=0; i<nbEdges; i++ )
	{
		einfo=&edgeInf[i];
		v0=_topology->getEdge(i)[0];
		v1=_topology->getEdge(i)[1];
		dp0=dx[v0];
		dp1=dx[v1];
		dp = dp1-dp0;
       
		df[v1]+= (einfo->DfDx*dp) * kFactor;
		df[v0]-= (einfo->DfDx.transposeMultiply(dp)) * kFactor;
	}
*/

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < size)
    {
		//v0=_topology->getEdge(i)[0];
        int v0 = (int) edgeInfo[i*EDGE_SIZE+V0];
        //v1=_topology->getEdge(i)[1];
        int v1 = (int) edgeInfo[i*EDGE_SIZE+V1];

        real dp0[3];
        real dp1[3];
        real dp[3];

        //dp0=dx[v0];
        dp0[0] = dx[v0*3];
        dp0[1] = dx[v0*3+1];
        dp0[2] = dx[v0*3+2];

        //dp1=dx[v1];
        dp1[0] = dx[v1*3];
        dp1[1] = dx[v1*3+1];
        dp1[2] = dx[v1*3+2];

        //dp = dp1-dp0;
        dp[0] = dp1[0] - dp0[0];
        dp[1] = dp1[1] - dp0[1];
        dp[2] = dp1[2] - dp0[2];

        const real* matrix = &edgeInfo[i*EDGE_SIZE+DFDX];

		//contribEdge[i] = -(einfo->DfDx.transposeMultiply(dp)) * kFactor;
        real transposeMultiplied[3];
        transposeMultiplied[0] = matrix[0]*dp[0] + matrix[3]*dp[1] + matrix[6]*dp[2];
        transposeMultiplied[1] = matrix[1]*dp[0] + matrix[4]*dp[1] + matrix[7]*dp[2];
        transposeMultiplied[2] = matrix[2]*dp[0] + matrix[5]*dp[1] + matrix[8]*dp[2];

        contribEdge[i*6] = - (transposeMultiplied[0]*kFactor);
        contribEdge[i*6+1] = - (transposeMultiplied[1]*kFactor) ;
        contribEdge[i*6+2] =  - (transposeMultiplied[2]*kFactor);

		//contribEdge[i] = (einfo->DfDx*dp) * kFactor;
        real multiplied[3];

        multiplied[0] = matrix[0]*dp[0] + matrix[1]*dp[1] + matrix[2]*dp[2];
        multiplied[1] = matrix[3]*dp[0] + matrix[4]*dp[1] + matrix[5]*dp[2];
        multiplied[2] = matrix[6]*dp[0] + matrix[7]*dp[1] + matrix[8]*dp[2];

        contribEdge[i*6+3] = (multiplied[0]*kFactor);
        contribEdge[i*6+4] = (multiplied[1]*kFactor);
        contribEdge[i*6+5] = (multiplied[2]*kFactor);
   }

}

template <class real>
__global__ void TetrahedralTensorMassForceFieldCuda_MemoryAlignment_PointsAccumulation_kernel(int nbPoints, int nbMaxEdgesPerNode, const real* contribEdge, const int* neighbourhoodPoints, real* f)
{
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int forceIndex = threadIndex / 3;
    int offset = threadIndex % 3;

    if(forceIndex < nbPoints)
    {
        int indexNeighbourhood = forceIndex * nbMaxEdgesPerNode;
        real acc = 0;

        for(int j=0; j<nbMaxEdgesPerNode; j++)
        {
            int indexEdge = neighbourhoodPoints[indexNeighbourhood+j];
            if(indexEdge == -1)
                break;
            else
            {
                acc += contribEdge[indexEdge*3+offset];
            }
        }

        f[threadIndex] += acc;
    }
}

template <class real>
__global__ void TetrahedralTensorMassForceFieldCuda_PointsAccumulation_kernel(int nbPoints, int nbMaxEdgesPerNode, const real* contribEdge, const int* neighbourhoodPoints, real* f)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < nbPoints)
    {
        int indexNeighbourhood = i*nbMaxEdgesPerNode;
        real acc[3];
        acc[0] = 0; acc[1] = 0; acc[2] = 0;

        for(int j=0; j<nbMaxEdgesPerNode; j++)
        {
            int indexEdge = neighbourhoodPoints[indexNeighbourhood+j];
            if(indexEdge == -1)
                break;
            else
            {
                acc[0] += contribEdge[indexEdge*3];
                acc[1] += contribEdge[indexEdge*3+1];
                acc[2] += contribEdge[indexEdge*3+2];
            }
        }

        f[i*3] += acc[0];
        f[i*3+1] += acc[1];
        f[i*3+2] += acc[2];
    }
    
}

//////////////////////
// CPU-side methods //
//////////////////////

void TetrahedralTensorMassForceFieldCuda3f_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo )
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbEdges+BSIZE-1)/BSIZE,1);

    TetrahedralTensorMassForceFieldCuda_addForce_EdgeContribution_kernel<float><<<grid, threads>>>(nbEdges, (float*)contribEdge, (const float*) x, (const float*) initialPoints, (const float*) edgeInfo);
    mycudaDebugError("TetrahedralTensorMassForceField_addForce_GPUGEMS_EdgeConstribution_kernel<float>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints*3+BSIZE-1)/BSIZE,1);

    TetrahedralTensorMassForceFieldCuda_MemoryAlignment_PointsAccumulation_kernel<float><<<grid2, threads2>>>(nbPoints, nbMaxEdgesPerNode, (const float*) contribEdge, (const int*) neighbourhoodPoints, (float*) f);
    mycudaDebugError("TetrahedralTensorMassForceFieldCuda_addForce_GPUGEMS_PointsAccumulation_kernel<float>");

    if(TIMING)
        cudaDeviceSynchronize();
}

void TetrahedralTensorMassForceFieldCuda3f_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx,  const void* edgeInfo, float kFactor)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbEdges+BSIZE-1)/BSIZE,1);
    
    TetrahedralTensorMassForceFieldCuda_addDForce_EdgeContribution_kernel<float><<<grid, threads>>>(nbEdges, (float*)contribEdge, (const float*) dx, (const float*) edgeInfo, kFactor);
    mycudaDebugError("TetrahedralTensorMassForceField_addForce_GPUGEMS_EdgeConstribution_kernel<float>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints*3+BSIZE-1)/BSIZE,1);

    TetrahedralTensorMassForceFieldCuda_MemoryAlignment_PointsAccumulation_kernel<float><<<grid2, threads2>>>(nbPoints, nbMaxEdgesPerNode, (const float*) contribEdge, (const int*) neighbourhoodPoints, (float*) df);
    mycudaDebugError("TetrahedralTensorMassForceFieldCuda_addForce_GPUGEMS_PointsAccumulation_kernel<float,1>");
    
    if(TIMING)
        cudaDeviceSynchronize();
}


#ifdef SOFA_GPU_CUDA_DOUBLE

void TetrahedralTensorMassForceFieldCuda3d_addForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* f, const void* x, const void* initialPoints, const void* edgeInfo )
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbEdges+BSIZE-1)/BSIZE,1);

	TetrahedralTensorMassForceFieldCuda_addForce_EdgeContribution_kernel<double><<<grid, threads>>>(nbEdges, (double*)contribEdge, (const double*) x, (const double*) initialPoints, (double*) edgeInfo);
    mycudaDebugError("TetrahedralTensorMassForceField_addForce_GPUGEMS_EdgeConstribution_kernel<double>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints*3+BSIZE-1)/BSIZE,1);

    TetrahedralTensorMassForceFieldCuda_MemoryAlignment_PointsAccumulation_kernel<double><<<grid2, threads2>>>(nbPoints, nbMaxEdgesPerNode, (const double*) contribEdge, (const int*) neighbourhoodPoints, (double*) f);
    mycudaDebugError("TetrahedralTensorMassForceFieldCuda_addForce_GPUGEMS_PointsAccumulation_kernel<double>");

    if(TIMING)
        cudaDeviceSynchronize();
}

void TetrahedralTensorMassForceFieldCuda3d_addDForce(int nbPoints, int nbMaxEdgesPerNode, const void* neighbourhoodPoints, void* contribEdge, int nbEdges, void* df, const void* dx,  const void* edgeInfo, double kFactor)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbEdges+BSIZE-1)/BSIZE,1);
    
    TetrahedralTensorMassForceFieldCuda_addDForce_EdgeContribution_kernel<double><<<grid, threads>>>(nbEdges, (double*)contribEdge, (const double*) dx, (const double*) edgeInfo, kFactor);
    mycudaDebugError("TetrahedralTensorMassForceField_addForce_GPUGEMS_EdgeConstribution_kernel<double>");

    dim3 threads2(BSIZE);
    dim3 grid2((nbPoints*3+BSIZE-1)/BSIZE,1);

    TetrahedralTensorMassForceFieldCuda_MemoryAlignment_PointsAccumulation_kernel<double><<<grid2, threads2>>>(nbPoints, nbMaxEdgesPerNode, (const double*) contribEdge, (const int*) neighbourhoodPoints, (double*) df);
    mycudaDebugError("TetrahedralTensorMassForceFieldCuda_addForce_GPUGEMS_PointsAccumulation_kernel<double,1>");
    
    if(TIMING)
        cudaDeviceSynchronize();
}
#endif


#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif


