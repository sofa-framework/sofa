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
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_CPP

#include <ArticulatedSystemPlugin/ArticulatedSystemMapping.inl>

#include <sofa/core/ObjectFactory.h>

namespace articulatedsystemplugin
{

using namespace sofa::defaulttype;

// Register in the Factory
void registerArticulatedSystemMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Mapping between a set of 6D DOF's and a set of angles (µ) using an articulated hierarchy container.")
    .add< ArticulatedSystemMapping< Vec1Types, Rigid3Types, Rigid3Types > >());
}

template class SOFA_ARTICULATEDSYSTEMPLUGIN_API ArticulatedSystemMapping< Vec1Types, Rigid3Types, Rigid3Types >;


} //namespace articulatedsystemplugin

