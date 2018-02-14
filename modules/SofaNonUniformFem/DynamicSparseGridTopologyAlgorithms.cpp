/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYALGORITHMS_CPP
#include <SofaNonUniformFem/DynamicSparseGridTopologyAlgorithms.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(DynamicSparseGridTopologyAlgorithms);
int DynamicSparseGridTopologyAlgorithmsClass = core::RegisterObject("Hexahedron set topology algorithms")
#ifdef SOFA_FLOAT
        .add< DynamicSparseGridTopologyAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< DynamicSparseGridTopologyAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< DynamicSparseGridTopologyAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< DynamicSparseGridTopologyAlgorithms<Vec2dTypes> >()
        .add< DynamicSparseGridTopologyAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DynamicSparseGridTopologyAlgorithms<Vec2fTypes> >()
        .add< DynamicSparseGridTopologyAlgorithms<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class DynamicSparseGridTopologyAlgorithms<Vec3dTypes>;
template class DynamicSparseGridTopologyAlgorithms<Vec2dTypes>;
template class DynamicSparseGridTopologyAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class DynamicSparseGridTopologyAlgorithms<Vec3fTypes>;
template class DynamicSparseGridTopologyAlgorithms<Vec2fTypes>;
template class DynamicSparseGridTopologyAlgorithms<Vec1fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

