/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "CudaTypes.h"
#include "CudaIdentityMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::gpu::cuda;

template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3fTypes> > >;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3dTypes> > >;
#endif

// template class  IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class  IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3f1Types> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3f1Types> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3dTypes> > >;
template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3fTypes> > >;
// template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3dTypes> > >;
template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3fTypes> > >;
template class  IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3f1Types> > >;

} // namespace mapping

} // namespace component

namespace gpu
{

namespace cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(CudaIdentityMapping)

int IdentityMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3fTypes> > > >()

#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3dTypes> > > >()
#endif

// .add< IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3f1Types> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3fTypes> > > >()
// .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3f1Types> > > >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
