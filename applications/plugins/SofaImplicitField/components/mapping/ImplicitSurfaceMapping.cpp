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
#define SOFA_COMPONENT_MAPPING_IMPLICITSURFACEMAPPING_CPP
#include <SofaImplicitField/config.h>
#include <sofa/core/ObjectFactory.h>
#include "ImplicitSurfaceMapping.inl"

namespace sofa::component::mapping
{

using namespace sofa::defaulttype;

// Register in the Factory
void registerImplicitSurfaceMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Compute an iso-surface from a set of particles.")
    .add< ImplicitSurfaceMapping< Vec3dTypes, Vec3dTypes > >());
}

template class SOFA_SOFAIMPLICITFIELD_API ImplicitSurfaceMapping< Vec3dTypes, Vec3dTypes >;


} // namespace sofa::component::mapping

