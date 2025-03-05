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
#define  SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaCUDA/component/mapping/linear/CudaIdentityMapping.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping::linear
{

using namespace sofa::defaulttype;
using namespace sofa::gpu::cuda;

// CudaVec3fTypes
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, Vec3Types>;
template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3Types, CudaVec3fTypes>;

// CudaVec3f1Types
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, Vec3Types>;
template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3dTypes, CudaVec3f1Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
// CudaVec3dTypes
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, Vec3Types>;

template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3Types, CudaVec3dTypes>;
#endif

} // namespace sofa::component::mapping::linear


namespace sofa::gpu::cuda
{

using namespace sofa::defaulttype;
using namespace sofa::component::mapping::linear;

int IdentityMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
    // CudaVec3fTypes
    .add< IdentityMapping< CudaVec3fTypes, CudaVec3fTypes> >()
    .add< IdentityMapping< CudaVec3fTypes, CudaVec3f1Types> >()
    .add< IdentityMapping< CudaVec3fTypes, Vec3Types> >()
    .add< IdentityMapping< Vec3Types, CudaVec3fTypes> >()
    
    // CudaVec3f1Types
    .add< IdentityMapping< CudaVec3f1Types, CudaVec3f1Types> >()
    .add< IdentityMapping< CudaVec3f1Types, CudaVec3fTypes> >()
    .add< IdentityMapping< CudaVec3f1Types, Vec3Types> >()
    .add< IdentityMapping< Vec3Types, CudaVec3f1Types> >()
          
#ifdef SOFA_GPU_CUDA_DOUBLE
    // CudaVec3dTypes
    .add< IdentityMapping< CudaVec3dTypes, CudaVec3dTypes> >()
    .add< IdentityMapping< CudaVec3dTypes, CudaVec3fTypes> >()
    .add< IdentityMapping< CudaVec3dTypes, Vec3Types> >()

    .add< IdentityMapping< CudaVec3fTypes, CudaVec3dTypes> >()
    .add< IdentityMapping< Vec3Types, CudaVec3dTypes> >()

#endif
    ;

} // namespace sofa::gpu::cuda
