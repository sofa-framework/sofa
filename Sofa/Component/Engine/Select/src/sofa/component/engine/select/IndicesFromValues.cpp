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
#define SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_CPP
#include <sofa/component/engine/select/IndicesFromValues.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::select
{

int IndicesFromValuesClass = core::RegisterObject("Find the indices of a list of values within a larger set of values")
        .add< IndicesFromValues<std::string> >()
        .add< IndicesFromValues<int> >()
        .add< IndicesFromValues<unsigned int> >()
        .add< IndicesFromValues< type::fixed_array<unsigned int, 2> > >()
        .add< IndicesFromValues< type::fixed_array<unsigned int, 3> > >()
        .add< IndicesFromValues< type::fixed_array<unsigned int, 4> > >()
        .add< IndicesFromValues< type::fixed_array<unsigned int, 8> > >()
        .add< IndicesFromValues<SReal> >()
        .add< IndicesFromValues<type::Vec2> >()
        .add< IndicesFromValues<type::Vec3> >()
        // .add< IndicesFromValues<defaulttype::Rigid2Types::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid2Types::Deriv> >()
        // .add< IndicesFromValues<defaulttype::Rigid3Types::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid3Types::Deriv> >()
 
        ;

template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<std::string>;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<int>;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<unsigned int>;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 2> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 3> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 4> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 8> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<SReal>;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<type::Vec2>;
template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<type::Vec3>;
// template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid2Types::Coord>;
// template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid2Types::Deriv>;
// template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid3Types::Coord>;
// template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid3Types::Deriv>;
 

} //namespace sofa::component::engine::select
