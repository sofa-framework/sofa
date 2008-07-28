/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

int TriangleSetTopologyModifierClass = core::RegisterObject("Triangle set topology modifier")
#ifndef SOFA_FLOAT
        .add< TriangleSetTopologyModifier<Vec3dTypes> >()
        .add< TriangleSetTopologyModifier<Vec2dTypes> >()
        .add< TriangleSetTopologyModifier<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangleSetTopologyModifier<Vec3fTypes> >()
        .add< TriangleSetTopologyModifier<Vec2fTypes> >()
        .add< TriangleSetTopologyModifier<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class TriangleSetTopologyModifier<Vec3dTypes>;
template class TriangleSetTopologyModifier<Vec2dTypes>;
template class TriangleSetTopologyModifier<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class TriangleSetTopologyModifier<Vec3fTypes>;
template class TriangleSetTopologyModifier<Vec2fTypes>;
template class TriangleSetTopologyModifier<Vec1fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa
