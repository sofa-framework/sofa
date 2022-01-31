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
#pragma once

#include <sofa/core/config.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaBaseMechanics/MassType.h>
#include <type_traits>

namespace sofa::component::mass
{

template<typename DataTypes> 
struct MassType<DataTypes,
                 std::enable_if_t < std::is_same_v<DataTypes, sofa::defaulttype::StdRigidTypes<DataTypes::spatial_dimensions, typename DataTypes::Real> > >
> 
{
    using type = sofa::defaulttype::RigidMass< DataTypes::spatial_dimensions, typename DataTypes::Real>;
};


} // namespace sofa::component::mass
