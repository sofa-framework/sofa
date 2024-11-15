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
#define SOFA_COMPONENT_MAPPING_HEXAHEDRONCOMPOSITEFEMMAPPING_CPP

#include <sofa/component/solidmechanics/fem/nonuniform/HexahedronCompositeFEMMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::solidmechanics::fem::nonuniform
{

using namespace defaulttype;
using namespace core;
using namespace core::behavior;

void registerHexahedronCompositeFEMMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Set the point to the center of mass of the DOFs it is attached to.")
        .add< HexahedronCompositeFEMMapping< Mapping< Vec3Types, Vec3Types > > >());
}

template class HexahedronCompositeFEMMapping< Mapping< Vec3Types, Vec3Types > >;

} // namespace sofa::component::solidmechanics::fem::nonuniform
