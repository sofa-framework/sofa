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
#ifndef SOFA_GPU_CUDA_CUDAMESHMATRIXMASS_CPP
#define SOFA_GPU_CUDA_CUDAMESHMATRIXMASS_CPP

#include <SofaCUDA/component/mass/CudaMeshMatrixMass.inl>
#include <sofa/component/mass/MeshMatrixMass.inl>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>

#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/QuadSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa
{

namespace component
{

namespace mass
{

template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, sofa::gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes, sofa::gpu::cuda::CudaVec2fTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1fTypes, sofa::gpu::cuda::CudaVec3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2dTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec2dTypes, sofa::gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes, sofa::gpu::cuda::CudaVec2dTypes>;
template class SOFA_GPU_CUDA_API MeshMatrixMass<sofa::gpu::cuda::CudaVec1dTypes, sofa::gpu::cuda::CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE


} // namespace mass

} // namespace component

namespace gpu
{

namespace cuda
{

int MeshMatrixMassClassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::mass::MeshMatrixMass<CudaVec3fTypes > >(true)
        .add< component::mass::MeshMatrixMass<CudaVec2fTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec2fTypes, CudaVec3fTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1fTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1fTypes, CudaVec2fTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1fTypes, CudaVec3fTypes > >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::mass::MeshMatrixMass<CudaVec3dTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec2dTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec2dTypes, CudaVec3dTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1dTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1dTypes, CudaVec2dTypes > >()
        .add< component::mass::MeshMatrixMass<CudaVec1dTypes, CudaVec3dTypes > >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif //SOFA_GPU_CUDA_CUDAMESHMATRIXMASS_CPP

