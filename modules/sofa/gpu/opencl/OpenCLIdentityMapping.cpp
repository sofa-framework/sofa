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
#include "OpenCLTypes.h"
#include "OpenCLIdentityMapping.inl"
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
using namespace sofa::gpu::opencl;

template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3fTypes> > >;

template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<OpenCLVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<OpenCLVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3dTypes> > >;

template class  IdentityMapping< Mapping< State<OpenCLVec3d1Types>, MappedModel<ExtVec3dTypes> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3dTypes>, MappedModel<ExtVec3dTypes> > >;

// template class  IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<OpenCLVec3f1Types> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<Vec3dTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<Vec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3f1Types> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<OpenCLVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<Vec3dTypes> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<Vec3fTypes> > >;
// template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<ExtVec3dTypes> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<ExtVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<OpenCLVec3fTypes> > >;
template class  IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3f1Types> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<OpenCLVec3fTypes> > >;
template class  IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<OpenCLVec3f1Types> > >;

} // namespace mapping

} // namespace component

namespace gpu
{

namespace opencl
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(OpenCLIdentityMapping)

int IdentityMappingOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3fTypes> > > >()

        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<OpenCLVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<OpenCLVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3dTypes> > > >()

        .add< IdentityMapping< Mapping< State<OpenCLVec3d1Types>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3dTypes>, MappedModel<ExtVec3fTypes> > > >()

// .add< IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<OpenCLVec3f1Types> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<OpenCLVec3f1Types> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<OpenCLVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<OpenCLVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<Vec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<Vec3fTypes> > > >()
// .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3f1Types>, MechanicalState<OpenCLVec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<OpenCLVec3fTypes>, MechanicalState<OpenCLVec3f1Types> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3f1Types>, MappedModel<OpenCLVec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<OpenCLVec3fTypes>, MappedModel<OpenCLVec3f1Types> > > >()
        ;

} // namespace opencl

} // namespace gpu

} // namespace sofa
