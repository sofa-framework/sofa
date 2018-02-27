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
#include "OpenCLTypes.h"
#include "OpenCLIdentityMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <SofaBaseMechanics/IdentityMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

#ifndef SOFA_FLOAT
namespace core
{
    // wtf is it needed by gcc to link? I suspect a namespace complex situation
    template class core::Mapping< defaulttype::Vec3dTypes, gpu::opencl::OpenCLVec3fTypes>;
}
#endif

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::opencl;

template class IdentityMapping< OpenCLVec3fTypes, OpenCLVec3fTypes>;
template class IdentityMapping< OpenCLVec3fTypes, Vec3fTypes>;
#ifndef SOFA_FLOAT
template class IdentityMapping< OpenCLVec3fTypes, Vec3dTypes>;
#endif
template class IdentityMapping< Vec3fTypes, OpenCLVec3fTypes>;
#ifndef SOFA_FLOAT
template class IdentityMapping< Vec3dTypes, OpenCLVec3fTypes>;
#endif

template class IdentityMapping< OpenCLVec3fTypes, OpenCLVec3dTypes>;
template class IdentityMapping< OpenCLVec3dTypes, OpenCLVec3fTypes>;
template class IdentityMapping< OpenCLVec3dTypes, OpenCLVec3dTypes>;
template class IdentityMapping< OpenCLVec3dTypes, Vec3fTypes>;
#ifndef SOFA_FLOAT
template class IdentityMapping< OpenCLVec3dTypes, Vec3dTypes>;
#endif
template class IdentityMapping< Vec3fTypes, OpenCLVec3dTypes>;
#ifndef SOFA_FLOAT
template class IdentityMapping< Vec3dTypes, OpenCLVec3dTypes>;
#endif

template class IdentityMapping< OpenCLVec3d1Types, ExtVec3fTypes>;
template class IdentityMapping< OpenCLVec3dTypes, ExtVec3fTypes>;

template class IdentityMapping< OpenCLVec3fTypes, ExtVec3fTypes>;
template class IdentityMapping< OpenCLVec3f1Types, OpenCLVec3f1Types>;
#ifndef SOFA_FLOAT
template class IdentityMapping< OpenCLVec3f1Types, Vec3dTypes>;
#endif
template class IdentityMapping< OpenCLVec3f1Types, Vec3fTypes>;
#ifndef SOFA_FLOAT
template class IdentityMapping< Vec3dTypes, OpenCLVec3f1Types>;
#endif
template class IdentityMapping< Vec3fTypes, OpenCLVec3f1Types>;
template class IdentityMapping< OpenCLVec3f1Types, ExtVec3fTypes>;
template class IdentityMapping< OpenCLVec3f1Types, OpenCLVec3fTypes>;
template class IdentityMapping< OpenCLVec3fTypes, OpenCLVec3f1Types>;

} // namespace mapping

} // namespace component

namespace gpu
{

namespace opencl
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(OpenCLIdentityMapping)

int IdentityMappingOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< IdentityMapping< OpenCLVec3fTypes, OpenCLVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3fTypes, Vec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< OpenCLVec3fTypes, Vec3dTypes> >()
#endif
        .add< IdentityMapping< Vec3fTypes, OpenCLVec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< Vec3dTypes, OpenCLVec3fTypes> >()
#endif

        .add< IdentityMapping< OpenCLVec3fTypes, OpenCLVec3dTypes> >()
        .add< IdentityMapping< OpenCLVec3dTypes, OpenCLVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3dTypes, OpenCLVec3dTypes> >()
        .add< IdentityMapping< OpenCLVec3dTypes, Vec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< OpenCLVec3dTypes, Vec3dTypes> >()
#endif
        .add< IdentityMapping< Vec3fTypes, OpenCLVec3dTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< Vec3dTypes, OpenCLVec3dTypes> >()
#endif

        .add< IdentityMapping< OpenCLVec3d1Types, ExtVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3dTypes, ExtVec3fTypes> >()

        .add< IdentityMapping< OpenCLVec3fTypes, ExtVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3f1Types, OpenCLVec3f1Types> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< OpenCLVec3f1Types, Vec3dTypes> >()
#endif
        .add< IdentityMapping< OpenCLVec3f1Types, Vec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< IdentityMapping< Vec3dTypes, OpenCLVec3f1Types> >()
#endif
        .add< IdentityMapping< Vec3fTypes, OpenCLVec3f1Types> >()
        .add< IdentityMapping< OpenCLVec3f1Types, ExtVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3f1Types, OpenCLVec3fTypes> >()
        .add< IdentityMapping< OpenCLVec3fTypes, OpenCLVec3f1Types> >()
        ;

} // namespace opencl

} // namespace gpu

} // namespace sofa
