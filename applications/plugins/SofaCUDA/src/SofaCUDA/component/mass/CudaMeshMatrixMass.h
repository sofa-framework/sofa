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
#ifndef SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H
#define SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/mass/MeshMatrixMass.h>


namespace sofa::component::mass
{

using namespace sofa::gpu::cuda;
using namespace sofa::component::mass;

template<class GeometricalTypes>
class MeshMatrixMassInternalData<CudaVec3fTypes,float, GeometricalTypes>
{
public:
    /// Cuda vector copying the vertex mass (enabling deviceRead)
    CudaVector<float> vMass;
};

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>::copyVertexMass();

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>::addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& /*vf*/, const DataVecCoord& /* */, const DataVecDeriv& /* */);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f);




template<class GeometricalTypes>
class MeshMatrixMassInternalData<CudaVec2fTypes, float, GeometricalTypes>
{
public:
    /// Cuda vector copying the vertex mass (enabling deviceRead)
    CudaVector<float> vMass;
};

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>::copyVertexMass();

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>::addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& /*vf*/, const DataVecCoord& /* */, const DataVecDeriv& /* */);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f);




template<class GeometricalTypes>
class MeshMatrixMassInternalData<CudaVec1fTypes, float, GeometricalTypes>
{
public:
    /// Cuda vector copying the vertex mass (enabling deviceRead)
    CudaVector<float> vMass;
};

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>::copyVertexMass();

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>::addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& /*vf*/, const DataVecCoord& /* */, const DataVecDeriv& /* */);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f);


#ifndef SOFA_GPU_CUDA_CUDAMESHMATRIXMASS_CPP
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes, sofa::gpu::cuda::CudaVec2fTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes, sofa::gpu::cuda::CudaVec3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2dTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2dTypes, sofa::gpu::cuda::CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes, sofa::gpu::cuda::CudaVec2dTypes>;
extern template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes, sofa::gpu::cuda::CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

#endif //SOFA_GPU_CUDA_CUDAMESHMATRIXMASS_CPP

} // namespace sofa::component::mass


#endif //SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H
