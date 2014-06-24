/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaNonUniformFem/HexahedronCompositeFEMMapping.inl>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
//#include <sofa/core/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(HexahedronCompositeFEMMapping)

using namespace defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int HexahedronCompositeFEMMappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< HexahedronCompositeFEMMapping< Mapping< Vec3dTypes, ExtVec3fTypes > > >()
        .add< HexahedronCompositeFEMMapping< Mapping< Vec3dTypes, Vec3dTypes    > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, ExtVec3fTypes > > >()
        .add< HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, Vec3fTypes    > > >()
#endif
//
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< HexahedronCompositeFEMMapping< Mapping<Vec3fTypes, Vec3dTypes > > >()
        .add< HexahedronCompositeFEMMapping< Mapping<Vec3dTypes, Vec3fTypes > > >()
#endif
#endif
        ;



#ifndef SOFA_FLOAT
template class HexahedronCompositeFEMMapping< Mapping< Vec3dTypes, ExtVec3fTypes >  >;
template class HexahedronCompositeFEMMapping< Mapping< Vec3dTypes, Vec3dTypes    >  >;
#endif
#ifndef SOFA_DOUBLE
template class HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, ExtVec3fTypes > >;
template class HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, Vec3fTypes    > >;
#endif

/*
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
	template class HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, ExtVec3fTypes > >;
	template class HexahedronCompositeFEMMapping< Mapping< Vec3fTypes, Vec3fTypes    > >;
#endif
#endif
*/



} // namespace mapping

} // namespace component

} // namespace sofa

