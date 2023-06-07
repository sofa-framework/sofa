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
#define SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_CPP
#include <sofa/component/engine/select/ValuesFromIndices.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::select
{

int ValuesFromIndicesClass = core::RegisterObject("Find the values given a list of indices")
        .add< ValuesFromIndices<std::string> >()
        .add< ValuesFromIndices<int> >()
        .add< ValuesFromIndices<unsigned int> >()
        .add< ValuesFromIndices< type::fixed_array<unsigned int, 2> > >()
        .add< ValuesFromIndices< type::fixed_array<unsigned int, 3> > >()
        .add< ValuesFromIndices< type::fixed_array<unsigned int, 4> > >()
        .add< ValuesFromIndices< type::fixed_array<unsigned int, 8> > >()
        .add< ValuesFromIndices<SReal> >()
        .add< ValuesFromIndices<type::Vec2> >()
        .add< ValuesFromIndices<type::Vec3> >()
        .add< ValuesFromIndices<type::Vec4> >()
        .add< ValuesFromIndices<type::Vec6> >()
        .add< ValuesFromIndices<defaulttype::Rigid2Types::Coord> >()
        .add< ValuesFromIndices<defaulttype::Rigid2Types::Deriv> >()
        .add< ValuesFromIndices<defaulttype::Rigid3Types::Coord> >()
        .add< ValuesFromIndices<defaulttype::Rigid3Types::Deriv> >()
 
        ;

template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<std::string>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<int>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<unsigned int>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 2> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 3> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 4> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 8> >;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<SReal>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec2>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec3>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec4>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec6>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid2Types::Coord>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid2Types::Deriv>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid3Types::Coord>;
template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid3Types::Deriv>;
 

} //namespace sofa::component::engine::select
