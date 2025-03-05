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
#define SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_CPP

#include <sofa/component/mapping/linear/IdentityMultiMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::defaulttype;

namespace sofa::component::mapping::linear
{

void registerIdentityMultiMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Concatenate several mechanical states together.")
        .add< IdentityMultiMapping< Vec3Types, Vec3Types > >()
        .add< IdentityMultiMapping< Rigid3Types, Rigid3Types > >());
}

template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMultiMapping< Vec3Types, Vec3Types >;
template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMultiMapping< Rigid3Types, Rigid3Types >;

} // namespace sofa::component::mapping::linear
