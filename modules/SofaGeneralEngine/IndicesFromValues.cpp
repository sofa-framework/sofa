/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaGeneralEngine/IndicesFromValues.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(IndicesFromValues)

int IndicesFromValuesClass = core::RegisterObject("Find the indices of a list of values within a larger set of values")
        .add< IndicesFromValues<std::string> >()
        .add< IndicesFromValues<int> >()
        .add< IndicesFromValues<unsigned int> >()
        .add< IndicesFromValues< helper::fixed_array<unsigned int, 2> > >()
        .add< IndicesFromValues< helper::fixed_array<unsigned int, 3> > >()
        .add< IndicesFromValues< helper::fixed_array<unsigned int, 4> > >()
        .add< IndicesFromValues< helper::fixed_array<unsigned int, 8> > >()
#ifndef SOFA_FLOAT
        .add< IndicesFromValues<double> >()
        .add< IndicesFromValues<defaulttype::Vec2d> >()
        .add< IndicesFromValues<defaulttype::Vec3d> >()
        // .add< IndicesFromValues<defaulttype::Rigid2dTypes::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid2dTypes::Deriv> >()
        // .add< IndicesFromValues<defaulttype::Rigid3dTypes::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid3dTypes::Deriv> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IndicesFromValues<float> >()
        .add< IndicesFromValues<defaulttype::Vec2f> >()
        .add< IndicesFromValues<defaulttype::Vec3f> >()
        // .add< IndicesFromValues<defaulttype::Rigid2fTypes::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid2fTypes::Deriv> >()
        // .add< IndicesFromValues<defaulttype::Rigid3fTypes::Coord> >()
        // .add< IndicesFromValues<defaulttype::Rigid3fTypes::Deriv> >()
#endif //SOFA_DOUBLE
        ;

template class SOFA_GENERAL_ENGINE_API IndicesFromValues<std::string>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<int>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<unsigned int>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 2> >;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 3> >;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 4> >;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues< helper::fixed_array<unsigned int, 8> >;
#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<double>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec2d>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec3d>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Coord>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2dTypes::Deriv>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Coord>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3dTypes::Deriv>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<float>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec2f>;
template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Vec3f>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Coord>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid2fTypes::Deriv>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Coord>;
// template class SOFA_GENERAL_ENGINE_API IndicesFromValues<defaulttype::Rigid3fTypes::Deriv>;
#endif //SOFA_DOUBLE

} // namespace constraint

} // namespace component

} // namespace sofa

