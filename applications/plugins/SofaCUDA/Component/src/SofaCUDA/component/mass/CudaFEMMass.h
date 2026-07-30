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
#pragma once

#include <SofaCUDA/component/config.h>

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/mass/FEMMass.h>

namespace sofa::component::mass
{

#ifndef SOFA_GPU_CUDA_CUDAFEMMASS_CPP
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Edge>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Triangle>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Quad>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Tetrahedron>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Hexahedron>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Prism>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3fTypes, sofa::geometry::Pyramid>;

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Edge>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Triangle>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Quad>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Tetrahedron>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Hexahedron>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Prism>;
extern template class SOFACUDA_COMPONENT_API FEMMass<sofa::gpu::cuda::CudaVec3dTypes, sofa::geometry::Pyramid>;
#endif
#endif

} // namespace sofa::component::mass
