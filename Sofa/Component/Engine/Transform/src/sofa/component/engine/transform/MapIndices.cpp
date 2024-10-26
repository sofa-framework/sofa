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
#define SOFA_COMPONENT_ENGINE_MAPINDICES_CPP
#include <sofa/component/engine/transform/MapIndices.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::transform
{

void registerMapIndices(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Apply a permutation to a set of indices.")
        .add< MapIndices<int> >()
        .add< MapIndices<unsigned int> >()
        .add< MapIndices< type::fixed_array<unsigned int, 2> > >()
        .add< MapIndices< type::fixed_array<unsigned int, 3> > >()
        .add< MapIndices< type::fixed_array<unsigned int, 4> > >()
        .add< MapIndices< type::fixed_array<unsigned int, 8> > >());
}

template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices<int>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices<unsigned int>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 2> >;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 3> >;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 4> >;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MapIndices< type::fixed_array<unsigned int, 8> >;

} //namespace sofa::component::engine::transform
