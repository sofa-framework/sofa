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
#include <sofa/component/mapping/SPHFluidSurfaceMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core::behavior;

SOFA_DECL_CLASS(SPHFluidSurfaceMapping)

// Register in the Factory
int SPHFluidSurfaceMappingClass = core::RegisterObject("TODO-SPHFluidSurfaceMappingClass")
        .addAlias("MarchingCubeMapping")
#ifndef SOFA_FLOAT
        .add< SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >()
// .add< SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >()
// .add< SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> >;
// template class SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
template class SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;
#endif
#ifndef SOFA_DOUBLE
template class SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
// template class SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SPHFluidSurfaceMapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> >;
#endif
#endif

// Mech -> Mapped

// Mech -> ExtMapped

} // namespace mapping

} // namespace component

} // namespace sofa

