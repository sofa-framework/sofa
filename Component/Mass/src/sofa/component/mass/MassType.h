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

#include <sofa/component/mass/config.h>

namespace sofa::component::mass
{
/*
 * This (empty) templated struct is used for determining a type of mass according to
 * the associated DataType. 
 * The generic version of it does not contain any type/definition,  
 * and will provoke an error if one is trying to determine a MassType without having
 * specialized this struct first.
 * For example, MassType specialized on Vec<N,Real> should return Real as its type.
 * (see VecMassType.h)
 * 
 * This is used by the Mass components to find a MassType according to their DataType.
 */
template<typename DataType>
struct MassType
{
    // if you want to associate a mass type YourType for a particular DataType
    // using type = YourType;
};


} // namespace sofa::component::mass
