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
#ifndef SOFA_GPU_CUDA_CUDASTANDARDTETRAHEDRALFEMFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASTANDARDTETRAHEDRALFEMFORCEFIELD_H

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaMiscFem/StandardTetrahedralFEMForceField.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{


template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3fTypes>::initNeighbourhoodPoints();

#ifdef SOFA_GPU_CUDA_DOUBLE
template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
void StandardTetrahedralFEMForceField<gpu::cuda::CudaVec3dTypes>::initNeighbourhoodPoints();
#endif


inline int& StandardTetrahedralFEMForceField_nbMaxTetraPerNode()
{
	static int nbMaxTetraPerNode = 0;
	return nbMaxTetraPerNode;
}

inline int& StandardTetrahedralFEMForceField_nbMaxTetraPerEdge()
{
	static int nbMaxTetraPerEdge = 0;
	return nbMaxTetraPerEdge;
}

// WARNING: we do the following because we cannot instantiate a CudaVector (calls cuda functions) during the load of the dynamic library on Windows (its a bug leading to a stale)

inline sofa::gpu::cuda::CudaVector<int>& StandardTetrahedralFEMForceField_neighbourhoodPoints()
{
	static sofa::gpu::cuda::CudaVector<int> neighbourhoodPoints;
	return neighbourhoodPoints;
}

inline sofa::gpu::cuda::CudaVector<float>& StandardTetrahedralFEMForceField_contribTetra()
{
	static sofa::gpu::cuda::CudaVector<float> contribTetra;
	return contribTetra;
}

inline sofa::gpu::cuda::CudaVector<int>& StandardTetrahedralFEMForceField_neighbourhoodEdges()
{
	static sofa::gpu::cuda::CudaVector<int> neighbourhoodEdges;
	return neighbourhoodEdges;
}

inline sofa::gpu::cuda::CudaVector<float>& StandardTetrahedralFEMForceField_contribDfDx()
{
	static sofa::gpu::cuda::CudaVector<float> contribDfDx;
	return contribDfDx;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //SOFA_GPU_CUDA_CUDASTANDARDTETRAHEDRALFEMFORCEFIELD_H
