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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP

#include "CudaTypes.h"
#include "CudaIdentityMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;

template class  IdentityMapping< CudaVec3fTypes, CudaVec3fTypes>;
#ifndef SOFA_DOUBLE
template class  IdentityMapping< CudaVec3fTypes, Vec3fTypes>;
template class  IdentityMapping< Vec3fTypes, CudaVec3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class  IdentityMapping< CudaVec3fTypes, Vec3dTypes>;
template class  IdentityMapping< Vec3dTypes, CudaVec3fTypes>;
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE
template class  IdentityMapping< CudaVec3fTypes, CudaVec3dTypes>;
template class  IdentityMapping< CudaVec3dTypes, CudaVec3fTypes>;
template class  IdentityMapping< CudaVec3dTypes, CudaVec3dTypes>;
template class  IdentityMapping< CudaVec3dTypes, Vec3fTypes>;
template class  IdentityMapping< CudaVec3dTypes, Vec3dTypes>;
#ifndef SOFA_DOUBLE
template class  IdentityMapping< Vec3dTypes, CudaVec3dTypes>;
#endif
#ifndef SOFA_FLOAT
template class  IdentityMapping< Vec3fTypes, CudaVec3dTypes>;
#endif

template class  IdentityMapping< CudaVec3d1Types, ExtVec3dTypes >;
template class  IdentityMapping< CudaVec3dTypes, ExtVec3dTypes >;
#endif
template class  IdentityMapping< CudaVec3f1Types, ExtVec3fTypes >;
template class  IdentityMapping< CudaVec3f1Types, CudaVec3f1Types>;
template class  IdentityMapping< CudaVec3f1Types, Vec3fTypes>;
#ifndef SOFA_FLOAT
template class  IdentityMapping< Vec3dTypes, CudaVec3f1Types>;
template class  IdentityMapping< CudaVec3f1Types, Vec3dTypes>;
template class  IdentityMapping< CudaVec3f1Types, ExtVec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class  IdentityMapping< Vec3fTypes, ExtVec3fTypes>;
#endif
template class  IdentityMapping< CudaVec3f1Types, CudaVec3fTypes>;
template class  IdentityMapping< CudaVec3fTypes, CudaVec3f1Types>;

} // namespace mapping

} // namespace component

namespace gpu
{

namespace cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(CudaIdentityMapping)

int IdentityMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< IdentityMapping< CudaVec3fTypes, CudaVec3fTypes> >()
        .add< IdentityMapping< CudaVec3fTypes, Vec3fTypes> >()
        .add< IdentityMapping< Vec3fTypes, CudaVec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< CudaVec3fTypes, Vec3dTypes> >()
        .add< IdentityMapping< Vec3dTypes, CudaVec3fTypes> >()
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< IdentityMapping< CudaVec3fTypes, CudaVec3dTypes> >()
        .add< IdentityMapping< CudaVec3dTypes, CudaVec3fTypes> >()
        .add< IdentityMapping< CudaVec3dTypes, CudaVec3dTypes> >()
        .add< IdentityMapping< CudaVec3dTypes, Vec3fTypes> >()
        .add< IdentityMapping< CudaVec3dTypes, Vec3dTypes> >()
        .add< IdentityMapping< Vec3fTypes, CudaVec3dTypes> >()
        .add< IdentityMapping< Vec3dTypes, CudaVec3dTypes> >()
        .add< IdentityMapping< CudaVec3d1Types, ExtVec3fTypes> >()
        .add< IdentityMapping< CudaVec3dTypes, ExtVec3fTypes> >()
#endif

        .add< IdentityMapping< CudaVec3fTypes, ExtVec3fTypes> >()
        .add< IdentityMapping< CudaVec3f1Types, CudaVec3f1Types> >()
        .add< IdentityMapping< CudaVec3f1Types, Vec3fTypes> >()
        .add< IdentityMapping< Vec3fTypes, CudaVec3f1Types> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< CudaVec3f1Types, Vec3dTypes> >()        
        .add< IdentityMapping< Vec3dTypes, CudaVec3f1Types> >()
#endif
        .add< IdentityMapping< CudaVec3f1Types, ExtVec3fTypes> >()
        .add< IdentityMapping< CudaVec3f1Types, CudaVec3fTypes> >()
        .add< IdentityMapping< CudaVec3fTypes, CudaVec3f1Types> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa


#endif // SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP
